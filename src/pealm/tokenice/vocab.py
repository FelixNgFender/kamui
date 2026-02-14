import dataclasses


@dataclasses.dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        itos = list(d["itos"])
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(
            stoi=stoi,
            itos=itos,
        )
