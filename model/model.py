from typing import Optional,Tuple,Union,List
import torch
from torch import nn
from transformers import PretrainedConfig
import math
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN

class MiniMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self,x):
        return x * torch.rsqrt(x.square().mean(-1,keepdim=True) + self.eps)
                               
    def forward(self,x):
        return self._norm(x.float()).type_as(x) * self.weight
    

def precompute_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    # RoPE-Step1：计算出来每一个小组的旋转频率
    freqs = 1.0 / rope_base ** (torch.arange(0,dim,2)[:dim//2].float() / dim)
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow = (
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",4),
            rope_scaling.get("beta_fast",4),
            rope_scaling.get("beta_slow",1),
        )

        # 只有当文本长度大于模型原有的上下文长度的时候，才进行YaRN插值
        if end > orig_max:
        # YaRN-Step1：划分区间，高频区不插值，高频区直接线性插值，中间地带平滑处理
            # YaRN核心公式 f(i)' = f(i) * ((1-ramp) + ramp / s)
            # ramp = 1 - 低频区 ramp = 0 -高频区，0~1之间就是中间地带

            # inv_dim是一个反向求解边界的公式，输入的是一个波长的阈值，输出的是一个频率索引位置(浮点数，0~dim//2-1)
            inv_dim = lambda b : (dim*math.log(orig_max / (b * 2 * math.pi))) / (2*math.log(rope_base))

            # 计算出高频区，中间地带，低频区的分界线 low and high
            low = max(math.floor(inv_dim(beta_fast)),0)
            high = min(math.ceil(inv_dim(beta_slow)),dim//2-1)

        # YaRN-Step2: 计算ramp
            ramp = torch.clamp((torch.arange(dim//2,device=freqs.device).float() - low) / max(high - low,0.001),0,1)

        # YaRN-Step3: 代入公式，计算新的频率：f(i)' = f(i) * ((1-ramp) + ramp / s)
            freqs = freqs * ((1-ramp) + ramp / factor)
    
    # RoPE-Step2：
    t = torch.arange(end,device=freqs.device)

    # 计算真正旋转的角度 m\theta, 通过outer运算，一下子将所有位置的旋转角度全部计算好了
    freqs = torch.outer(t,freqs).float()

    # 生成正弦和余弦
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1) * attn_factor

    return freqs_cos,freqs_sin

def apply_rotate_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    def rotate_half(x):
        # [a,b,c,d] --> [-c,-d,a,b]
        x1 = -x[...,x.shape[-1]//2:]
        x2 = x[...,:x.shape[-1]//2]

        return torch.cat([x1,x2],dim=-1)
    
    # q,k的维度一般是[batch_size,n_heads,seq_length,head_dim]
    # cos和sin的维度是 [batch_size,seq_length,head_dim] 缺失了n_heads
    q_embed = q * cos.unsqueeze(dim=unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(dim=unsqueeze_dim) 
    k_embed = k * cos.unsqueeze(dim=unsqueeze_dim)  + rotate_half(k) * sin.unsqueeze(dim=unsqueeze_dim)

    return q_embed,k_embed



def repeat_kv(x: torch.Tensor, n_rep: int)-> torch.Tensor:
    batch_size,seq_len,kv_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return(
            # step1: x[:,:,:,None,:]插入一个新维度 (B,S,2,128) -> (B,S,2,1,128)
            # step2: .expand(...,n_rep,...) 将刚才的 1 维度扩展为 n_rep(也就是 4) (B,S,2,128) -> (B,S,2,4,128)
            # step3: .reshape(...,kv_value_heads*n_rep,...) 将这两个维度合并，得到(B,S,8,128)
            x[:,:,:,None,:].expand(batch_size,seq_len,kv_value_heads,n_rep,head_dim).reshape(batch_size,seq_len,kv_value_heads*n_rep,head_dim)
        )

class Attention(nn.Module):
    def __init__(self,args:MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # attention_heads是指注意力头的数量，而num_key_value_heads是指键值头的数量
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 定义线性层的 QKV
        self.q_proj = nn.Linear(args.hidden_size,args.hidden_size,bias=False)
        self.k_proj = nn.Linear(args.hidden_size,self.head_dim*self.num_key_value_heads,bias=False)
        self.v_proj = nn.Linear(args.hidden_size,self.head_dim*self.num_key_value_heads,bias=False)
        self.o_proj = nn.Linear(args.hidden_size,args.hidden_size,bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional,"scaled_dot_product_attention") and args.flash_attention 

    def forward(
        self,
        x:torch.Tensor,
        position_embeddings:Tuple[torch.Tensor,torch.Tensor],
        past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
        use_cache=False,
        attention_mask:Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        batch_size,seq_len,_ = x.shape
        # 投影，计算q,k,v
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 使用view方法拆分多头
        xq = xq.view(batch_size,seq_len,self.n_local_heads,self.head_dim)
        xk = xk.view(batch_size,seq_len,self.num_key_value_heads,self.head_dim)
        xv = xv.view(batch_size,seq_len,self.num_key_value_heads,self.head_dim)

        # 对于 q,k 使用 RoPE 位置编码
        cos,sin = position_embeddings
        xq,xk = apply_rotate_pos_emb(xq,xk,cos,sin)

        # 使用 kv_cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 这里dim = 1是指在 seq_len 这个维度进行拼接
            xk = torch.cat([past_key,xk],dim=1)
            xv = torch.cat([past_value,xv],dim=1)
        past_kv = (xk,xv) if use_cache else None

        xq,xk,xv = (
            # x原来的维度：（B,S,H,D）
            # 这里transpose(1,2)将H和S维度交换，变成（B,H,S,D）
            # 为什么要交换维度：Pytorch 的矩阵乘法是在 seq_len 和 head_dim 维度上进行的，在 PyTorch 中，当你使用 @（矩阵乘法）操作两个 4 维张量时，它会遵循一个规则：只对最后两个维度进行矩阵乘法，前面的维度全部视为“批次（Batch）”维度。
            xq.transpose(1,2),
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2),
        )

        # attention计算
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask==1)):
            output = F.scaled_dot_product_attention(xq,xk,xv,dropout=self.dropout if self.training else 0.0,is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            # 这段代码在prefill阶段和解码阶段表现不一样，在解码阶段由于KV Cache，实际上不需要masked
            scores[:,:,:,-seq_len:] += torch.triu(torch.full((seq_len,seq_len),float("-inf"),device=scores.device),diagonal=1)

            if attention_mask is not None:
                # 处理一个batch的句子的时候，需要padding短的句子，通过加入attention_mask让模型忽略这些padding
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 广播机制是的张量可以和常数直接计算
                extended_attention_mask = (1.0-extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)

            output = scores @ xv

        output = output.transpose(1,2).reshape(batch_size,seq_len,-1)
        output = self.resid_dropout(self.o_proj(output))

        return output,past_kv

            
class FeedForward(nn.Module):
    def __init__(self,args:MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * (8.0 / 3))
            # 向上取整为 64 的整数倍，能够更适配硬件从而达到加速的效果
            intermediate_size = ((intermediate_size+64-1) // 64) * 64
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = args.intermediate_size
        # 扩展投影
        self.up_proj = nn.Linear(args.hidden_size,self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj = nn.Linear(args.hidden_size,self.intermediate_size,bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self,x):
        return self.dropout(self.down_proj((self.act_fn(self.gate_proj(x))*self.up_proj(x))))
        
class MOEFeedForward(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size,config.num_experts,bias=False)
        self.experts = nn.ModuleList([FeedForward(config,intermediate_size = config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self,x):
        batch_size,seq_len,hidden_dim = x.shape
        # 展平
        x_flat = x.view(-1,hidden_dim)
        # 计算每个token经过gate之后得到的分数
        # 维度：num_tokens x num_experts
        scores = F.softmax(self.gate(x_flat),dim=-1)
        # 取前 top-k 个作为 expert
        topk_weight,topk_idx = torch.topk(scores,k=self.config.num_experts_per_tok,dim=-1,sorted=False)
        # 对权重进行归一化
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1,keepdim=True)+1e-20)
        # 定义一个MOE输出结果的画布，之后在这上面填充结果
        y = torch.zeros_like(x_flat)

        for i,expert in enumerate(self.experts):
            # 确定对于当前的 expert，有哪些 token 需要被这个 expert 来处理
            mask = (topk_idx == i)
            if mask.意思any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1,1)
                # 这里 0的是 dim=0这个维度的下标和 token_idx 进行匹配
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx,self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size,seq_len,hidden_dim)
            

# 拼接一个 Transformer Block ： 包含了自注意力层和前馈神经网络层
class MiniMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 定义组件
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)
    
    def forward(self,hidden_states,position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        residual = hidden_states
        hidden_states,past_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        
        return hidden_states,past_key_value


class MiniMindModel(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size,self.num_hidden_layers = config.vocab_size,config.num_hidden_layers

        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # transformer 隐藏层的集合
        self.layers = nn.ModuleList(
            [MiniMindBlock(l,config) for l in range(self.num_hidden_layers)]
        )
        # 归一化层
        self.norm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        freqs_cos,freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            max_position_embeddings = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling
        )

        # register_buffer 用来在模型中保存一些“状态”，但这些状态不是模型的参数（不需要被优化器更新，不需要计算梯度）。
        # 如果直接赋值，当你把模型转移到 GPU (model.cuda()) 时，这个张量可能会被遗漏。用 register_buffer 注册后，PyTorch 就会把它当做模型的一部分，随模型一起移动设备。
        self.register_buffer("freqs_cos",freqs_cos, persistent=False)
        self.register_buffer("freqs_sin",freqs_sin, persistent=False)
    
    def forward(self,
        input_ids:torch.LongTensor = None,
        attention_mask:torch.FloatTensor = None,
        past_key_values:List[torch.FloatTensor] = None,
        use_cache:bool = False,
        **kwargs
    ):
        batch_size,seq_length = input_ids.shape

        if hasattr(past_key_values,'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # past_key_values的形状为：(num_layers,2,batch_size,num_heads,seq_length,head_dim)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos+seq_length],
            self.freqs_sin[start_pos:start_pos+seq_length]
        )

        presents = []

        for layer_idx, (layer,past_key_value) in enumerate(zip(self.layers,past_key_values)):
            hidden_states,present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents

# PreTrainedModel: 这是 Hugging Face 库中的一个核心基类。继承它，意味着你的模型自动拥有了加载预训练权重 (.from_pretrained()) 和保存模型 (.save_pretrained()) 等极其强大的功能。
# GenerationMixin: 这个类赋予了模型生成文本的能力。因为有了它，你之后就可以直接调用 model.generate() 方法来让大模型自己输出一段话，它内部封装了各种解码策略（如贪婪搜索、束搜索 Beam Search、Top-p 采样等）
class MiniMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self,config:MiniMindConfig):
        self.config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
        # 权重共享：编码与解码是互逆的过程，从语义上面来说很合理，而且能节省巨大的显存占用
        # embedding 层是正着存放，然后 Linear 是反着存放（为了矩阵运算更快）
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self,
                input_ids:Optional[torch.Tensor] = None,
                attention_mask:Optional[torch.Tensor] = None,
                # 训练时的标准答案，用于计算损失
                labels:Optional[torch.Tensor] = None,
                past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache: bool = False,
                # 可以是整数或者tensor
                logits_to_keep: Union[int,torch.Tensor] = 0,
                **kwargs):
        hidden_states,past_key_values = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **kwargs
        )
        # slice(start, stop) 是 Python 内置的切片对象。平时我们写的列表切片 my_list[-1:]，在底层其实就是 my_list[slice(-1, None)]。
        # 这行代码的意思是：如果传入的 logits_to_keep 是个整数（比如 1），那我就创建一个切片对象 slice(-1, None)。如果它本身就是个张量（比如高级的索引掩码），那我就直接用它。
        # 假设 logits_to_keep = 1，那么 slice_indices 现在就等价于 [-1:]。
        slice_indices = slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else logits_to_keep

        # 只取最后 logits_to_keep 个 token 的 logits
        # hidden_state的形状是(batch_size,seq_length,hidden_size)
        logits = self.lm_head(hidden_states[:,slice_indices,:])

        loss = None

        if labels is not None:
            shift_logits = logits[:,:-1,:].contiguous()
            shift_labels = labels[:,1:].contiguous()

            loss = F.cross_entropy(
                # cross_entropy的格式要求：(batch_size*seq_length,num_classes)
                shift_logits.view(-1,shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values
        )

        return output






    

