#/legal_dc_baseline/generation.py

"""A local, configurable replacement for Legal-DC's external Chinese generators."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LocalGenerator:
    def __init__(self, model_name: str, max_new_tokens: int, device: str | None,
                 local_files_only: bool = False) -> None:
        """Load an encoder-decoder generator without the deprecated pipeline alias.

        Transformers 5 removed the ``text2text-generation`` pipeline task.  Loading
        FLAN-T5 through its explicit AutoModel API works in both Transformers 4 and 5.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        configured_limit = getattr(self.model.config, "n_positions", None)
        configured_limit = configured_limit or getattr(self.model.config, "max_position_embeddings", None)
        self.max_input_tokens = int(configured_limit or 512)

    @staticmethod
    def prompt(
        query: str,
        passages: list[dict],
        *,
        insufficient_evidence_message: str | None = None,
    ) -> str:
        evidence = "\n\n".join(
            f"[{item['rank']}] {item['article_reference']}: {item['text']}" for item in passages
        )
        if insufficient_evidence_message:
            guard = (
                "Answer only from the supplied Constitution passages. Do not use outside "
                "knowledge. If the passages do not contain sufficient evidence, reply exactly: "
                f"{insufficient_evidence_message}"
            )
        else:
            guard = (
                "Answer only from the supplied Constitution passages. "
                "If the passages do not support an answer, say so."
            )
        return (
            f"You are a legal consultation assistant. {guard} Give a concise answer.\n\n"
            f"Passages:\n{evidence}\n\nQuestion: {query}\nAnswer:"
        )

    def answer(
        self,
        query: str,
        passages: list[dict],
        *,
        insufficient_evidence_message: str | None = None,
    ) -> str:
        inputs = self.tokenizer(
            self.prompt(query, passages, insufficient_evidence_message=insufficient_evidence_message),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
