from torch import nn
import clip
import torch


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()

        ctx_init = "This is a person re-identification task."
        ctx_dim = 512
        # use given words to initialize context vectors
        # ctx_init = ctx_init.replace("_", " ")
        # n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = embedding  # torch.Tensor

        # n_cls_ctx = 4
        # cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        # nn.init.normal_(cls_vectors, std=0.02)
        # self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        # self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        # self.num_class = num_class
        # self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts