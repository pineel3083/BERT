class BertConfig:
    """Small BERT/DCIM configuration object.

    The names mirror the common Hugging Face BERT knobs so the encoder shape is
    easy to recognize, while the dcim_* knobs describe the TP-DCIM dataflow from
    the paper at a software-simulation level.
    """

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dcim_block_size=256,
        dcim_qk_mode="viha",
        dcim_av_mode="a_stationary",
        dcim_tcs_thresholds=None,
        dcim_collect_stats=False,
    ):
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if dcim_qk_mode not in ("viha", "hiva"):
            raise ValueError("dcim_qk_mode must be either 'viha' or 'hiva'")
        if dcim_av_mode not in ("a_stationary", "v_stationary"):
            raise ValueError(
                "dcim_av_mode must be either 'a_stationary' or 'v_stationary'"
            )
        if dcim_block_size <= 0:
            raise ValueError("dcim_block_size must be positive")
        if dcim_tcs_thresholds is not None and len(dcim_tcs_thresholds) != 8:
            raise ValueError("dcim_tcs_thresholds must contain 8 MSB-to-LSB values")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

        self.dcim_block_size = dcim_block_size
        self.dcim_qk_mode = dcim_qk_mode
        self.dcim_av_mode = dcim_av_mode
        self.dcim_tcs_thresholds = dcim_tcs_thresholds
        self.dcim_collect_stats = dcim_collect_stats

    @property
    def attention_head_size(self):
        return self.hidden_size // self.num_attention_heads

    def to_dict(self):
        return dict(self.__dict__)
