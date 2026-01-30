#0.3B参数

@dataclass
class ModelArgs(PretrainedConfig):
    """
    3B - Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 增加kwargs.get以兼容代码中的属性引用
        self.max_batch_size = kwargs.get("max_batch_size", 8)
        self.max_seq_len = kwargs.get("max_seq_len", 4096)
        self.dtype = kwargs.get("dtype", "bf16")
        self.vocab_size = kwargs.get("vocab_size", 128000)
        self.dim = kwargs.get("dim", 1024)
        self.inter_dim = kwargs.get("inter_dim", 2048)
        self.moe_inter_dim = kwargs.get("moe_inter_dim", 512)
        self.n_layers = kwargs.get("n_layers", 8)
        self.n_dense_layers = kwargs.get("n_dense_layers", 1)
        self.n_heads = kwargs.get("n_heads", 8)
        # MoE
        self.n_routed_experts = kwargs.get("n_routed_experts", 4) #必须是GPU倍数
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.n_activated_experts = kwargs.get("n_activated_experts", 2)
        self.n_expert_groups = kwargs.get("n_expert_groups", 1)
        self.n_limited_groups = kwargs.get("n_limited_groups", 1)
        self.score_func = kwargs.get("score_func", "softmax")
        self.route_scale = kwargs.get("route_scale", 1.0)
        # MLA
        self.q_lora_rank = kwargs.get("q_lora_rank", 0)
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 256)
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", 64)
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 32)
        self.v_head_dim = kwargs.get("v_head_dim", 64)
        # YARN
        self.original_seq_len = kwargs.get("original_seq_len", 4096) #必须和max_seq_len一致
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_factor = kwargs.get("rope_factor", 40)
        self.beta_fast = kwargs.get("beta_fast", 32)
        self.beta_slow = kwargs.get("beta_slow", 1)
        self.mscale = kwargs.get("mscale", 1.0)
        # MTP
        self.n_mtp_depths = kwargs.get("n_mtp_depths", 2)  # ✅ MTP 预测深度


#1B参数

@dataclass
class ModelArgs(PretrainedConfig):
    """
    1B - Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 增加kwargs.get以兼容代码中的属性引用
        self.max_batch_size = kwargs.get("max_batch_size", 8)
        self.max_seq_len = kwargs.get("max_seq_len", 4096)
        self.dtype = kwargs.get("dtype", "bf16")
        self.vocab_size = kwargs.get("vocab_size", 51200)
        self.dim = kwargs.get("dim", 2048)
        self.inter_dim = kwargs.get("inter_dim", 8192)
        self.moe_inter_dim = kwargs.get("moe_inter_dim", 1024)
        self.n_layers = kwargs.get("n_layers", 24)
        self.n_dense_layers = kwargs.get("n_dense_layers", 1)
        self.n_heads = kwargs.get("n_heads", 16)
        # MoE
        self.n_routed_experts = kwargs.get("n_routed_experts", 12)
        self.n_shared_experts = kwargs.get("n_shared_experts", 2)
        self.n_activated_experts = kwargs.get("n_activated_experts", 4)
        self.n_expert_groups = kwargs.get("n_expert_groups", 1)
        self.n_limited_groups = kwargs.get("n_limited_groups", 1)
        self.score_func = kwargs.get("score_func", "softmax")
        self.route_scale = kwargs.get("route_scale", 1.0)
        # MLA
        self.q_lora_rank = kwargs.get("q_lora_rank", 0)
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 512)
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", 128)
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 64)
        self.v_head_dim = kwargs.get("v_head_dim", 128)
        # YARN
        self.original_seq_len = kwargs.get("original_seq_len", 4096)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_factor = kwargs.get("rope_factor", 40)
        self.beta_fast = kwargs.get("beta_fast", 32)
        self.beta_slow = kwargs.get("beta_slow", 1)
        self.mscale = kwargs.get("mscale", 1.0)
        # MTP
        self.n_mtp_depths = kwargs.get("n_mtp_depths", 4)  # ✅ MTP 预测深度

# 0.5B配置

@dataclass
class ModelArgs(PretrainedConfig):
    """
    1B - Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 增加kwargs.get以兼容代码中的属性引用
        self.max_batch_size = kwargs.get("max_batch_size", 8)
        self.max_seq_len = kwargs.get("max_seq_len", 4096)
        self.dtype = kwargs.get("dtype", "bf16")
        self.vocab_size = kwargs.get("vocab_size", 51200)
        self.dim = kwargs.get("dim", 1024) 
        self.inter_dim = kwargs.get("inter_dim", 4096)  
        self.moe_inter_dim = kwargs.get("moe_inter_dim", 512) #MoE的inter_dim
        self.n_layers = kwargs.get("n_layers", 12) 
        self.n_dense_layers = kwargs.get("n_dense_layers", 1)
        self.n_heads = kwargs.get("n_heads", 8) 
        # MoE
        self.n_routed_experts = kwargs.get("n_routed_experts", 6)  
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)  
        self.n_activated_experts = kwargs.get("n_activated_experts", 2) 
        self.n_expert_groups = kwargs.get("n_expert_groups", 1)
        self.n_limited_groups = kwargs.get("n_limited_groups", 1)
        self.score_func = kwargs.get("score_func", "softmax")
        self.route_scale = kwargs.get("route_scale", 1.0)
        # MLA
        self.q_lora_rank = kwargs.get("q_lora_rank", 0)
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 256)  
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", 64)  
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 32)  
        self.v_head_dim = kwargs.get("v_head_dim", 64)
        # YARN
        self.original_seq_len = kwargs.get("original_seq_len", 4096)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_factor = kwargs.get("rope_factor", 40)
        self.beta_fast = kwargs.get("beta_fast", 32)
        self.beta_slow = kwargs.get("beta_slow", 1)
        self.mscale = kwargs.get("mscale", 1.0)
        # MTP
        self.n_mtp_depths = kwargs.get("n_mtp_depths", 2) 
