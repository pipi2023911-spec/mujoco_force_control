import json
import pandas as pd

from experiments.c1_contact_feasibility import run


def test_contact_envelope_forward_inverse(tmp_path):
    output = run(tmp_path)
    table = pd.read_csv(output / "force_envelope.csv")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    assert table["臂额外广义力范数"].max() == 0.
    assert table["正逆差_N"].max() < 1e-7
    assert table["位置命令有效"].all()
    edge = table[table["压入量_m"] == 1e-8].set_index("切向速度_m每s")
    assert edge.loc[.00625, "逆解法向力_N"] > 5.
    assert edge.loc[.00234375, "逆解法向力_N"] < 5.
    assert abs(edge.loc[.00625, "逆解法向力_N"]-edge.loc[-.00625, "逆解法向力_N"]) < 1e-6
