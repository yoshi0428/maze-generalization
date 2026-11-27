"""Draw the architecture figures for the slides (BasicBlock, FeedForwardNet, DTNet).

Writes basic_block.svg, feedforward.svg and dtnet.svg next to this file. If `rsvg-convert` is
installed, it also writes a PNG of each figure at 2x scale.

    python docs/figures/make_figures.py
"""

import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).parent
PNG_ZOOM = 2

DEFS = """<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1f3a93"/></marker>
<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#c26a00"/></marker>
<style>
text{font-family:Helvetica,Arial,sans-serif;fill:#1a1a1a}
.box{fill:#eef2fb;stroke:#1f3a93;stroke-width:2}
.out{fill:#fdecea;stroke:#c0392b;stroke-width:2}
.outt{fill:#c0392b}
.op{fill:#fff;stroke:#1f3a93;stroke-width:2}
.ln{fill:none;stroke:#1f3a93;stroke-width:2}
.ar{fill:none;stroke:#1f3a93;stroke-width:2;marker-end:url(#a)}
.dash{fill:none;stroke:#1f3a93;stroke-width:2;stroke-dasharray:7 5;marker-end:url(#a)}
.grp{fill:#f7f9fd;stroke:#8a94a8;stroke-width:1.5;stroke-dasharray:8 6}
.rc{fill:none;stroke:#c26a00;stroke-width:2.5;marker-end:url(#b)}
.rcbox{fill:#fff4e0;stroke:#c26a00;stroke-width:2}
text.rct{fill:#c26a00}
.t{font-size:18px;text-anchor:middle}
.s{font-size:14px;fill:#555}
.sm{font-size:14px;fill:#555;text-anchor:middle}
.h{font-size:24px;font-weight:bold}
</style></defs>"""


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f"{DEFS}\n"
        f'<rect width="{width}" height="{height}" fill="#fff"/>\n'
        f"{body.strip()}\n"
        "</svg>\n"
    )


BASIC_BLOCK = svg(560, 640, """
<text x="30" y="44" class="h">BasicBlock</text>
<text x="30" y="68" class="s">residual block, no batchnorm</text>
<!-- output -->
<text x="380" y="112" class="t">out = ReLU(x + F(x))</text>
<rect x="330" y="150" width="100" height="40" rx="6" class="box"/><text x="380" y="176" class="t">ReLU</text>
<path d="M380,150 L380,124" class="ar"/>
<circle cx="380" cy="250" r="18" class="op"/><path d="M370,250 h20 M380,240 v20" class="ln"/>
<path d="M380,232 L380,192" class="ar"/>
<!-- residual branch -->
<rect x="90" y="300" width="180" height="50" rx="6" class="box"/><text x="180" y="324" class="t">conv2</text><text x="180" y="342" class="sm">3×3, 128→128</text>
<rect x="90" y="410" width="180" height="50" rx="6" class="box"/><text x="180" y="434" class="t">conv1</text><text x="180" y="452" class="sm">3×3, 128→128</text>
<path d="M180,410 L180,352" class="ar"/><text x="192" y="386" class="s">ReLU</text>
<path d="M180,300 L180,250 L360,250" class="ar"/><text x="130" y="270" style="font-size:18px">F(x)</text>
<!-- skip -->
<circle cx="380" cy="530" r="4" fill="#1f3a93"/>
<path d="M380,530 L180,530 L180,462" class="ar"/>
<path d="M380,530 L380,270" class="ar"/><text x="392" y="400" class="s">skip (identity)</text>
<path d="M380,590 L380,530" class="ln"/>
<text x="380" y="612" class="t">x : (B, 128, H, W)</text>
""")

FEEDFORWARD = svg(640, 800, """
<text x="30" y="44" class="h">FeedForwardNet</text>
<text x="30" y="68" class="s">no loop, depth fixed at 30 blocks</text>
<rect x="190" y="90" width="180" height="44" rx="6" class="out"/><text x="280" y="118" class="t outt">logits (B, 2, H, W)</text>
<rect x="190" y="170" width="180" height="50" rx="6" class="box"/><text x="280" y="194" class="t">head</text><text x="280" y="212" class="sm">conv 128→32→8→2</text>
<path d="M280,170 L280,136" class="ar"/>
<rect x="150" y="260" width="260" height="360" rx="10" class="grp"/>
<rect x="190" y="290" width="180" height="44" rx="6" class="box"/><text x="280" y="318" class="t">BasicBlock #30</text>
<text x="280" y="420" class="t" style="font-size:26px">⋮</text>
<rect x="190" y="470" width="180" height="44" rx="6" class="box"/><text x="280" y="498" class="t">BasicBlock #2</text>
<rect x="190" y="560" width="180" height="44" rx="6" class="box"/><text x="280" y="588" class="t">BasicBlock #1</text>
<path d="M280,560 L280,516" class="ar"/><path d="M280,470 L280,432" class="ar"/><path d="M280,392 L280,336" class="ar"/>
<path d="M280,290 L280,222" class="ar"/>
<text x="428" y="300" class="t" style="text-anchor:start">30 blocks</text>
<text x="428" y="324" class="s">own weights each,</text>
<text x="428" y="344" class="s">each runs once</text>
<rect x="190" y="660" width="180" height="50" rx="6" class="box"/><text x="280" y="684" class="t">projection</text><text x="280" y="702" class="sm">conv 3→128 + ReLU</text>
<path d="M280,660 L280,606" class="ar"/>
<circle cx="280" cy="760" r="20" class="op"/><text x="280" y="766" class="t">x</text>
<text x="310" y="766" class="s">(B, 3, H, W)</text>
<path d="M280,740 L280,712" class="ar"/>
""")

DTNET = svg(900, 1000, """
<text x="30" y="44" class="h">DTNetRecall</text>
<text x="30" y="68" class="s">DTNet with recall, forward pass</text>
<rect x="310" y="40" width="180" height="44" rx="6" class="out"/><text x="400" y="68" class="t outt">logits (B, 2, H, W)</text>
<rect x="310" y="120" width="180" height="50" rx="6" class="box"/><text x="400" y="144" class="t">head</text><text x="400" y="162" class="sm">conv 128→32→8→2</text>
<path d="M400,120 L400,86" class="ar"/>
<rect x="330" y="206" width="140" height="44" rx="6" class="box"/><text x="400" y="234" class="t">h (final)</text>
<path d="M400,206 L400,172" class="ar"/>
<rect x="600" y="206" width="140" height="44" rx="6" class="out"/><text x="670" y="234" class="t outt">state h</text>
<path d="M470,228 L598,228" class="ar"/>
<text x="412" y="278" class="s">after the last iteration</text>
<!-- recurrent group -->
<rect x="150" y="290" width="670" height="444" rx="10" class="grp"/>
<text x="170" y="320" style="font-size:18px;font-weight:bold">Recurrent block</text>
<text x="170" y="342" class="s">runs iters times, same weights</text>
<path d="M400,330 L400,252" class="ar"/>
<circle cx="400" cy="330" r="4" fill="#1f3a93"/>
<rect x="310" y="360" width="180" height="44" rx="6" class="box"/><text x="400" y="388" class="t">BasicBlock #2</text>
<path d="M400,360 L400,332" class="ln"/>
<rect x="310" y="434" width="180" height="44" rx="6" class="box"/><text x="400" y="462" class="t">BasicBlock #1</text>
<path d="M400,434 L400,406" class="ar"/>
<rect x="350" y="508" width="100" height="36" rx="6" class="box"/><text x="400" y="532" class="t">ReLU</text>
<path d="M400,508 L400,480" class="ar"/>
<!-- recall step: concat [h, x], then recall_conv -->
<rect x="290" y="560" width="240" height="160" rx="10" class="rcbox"/>
<text x="280" y="590" class="rct" style="font-size:18px;font-weight:bold;text-anchor:end">recall</text>
<text x="280" y="612" class="s rct" style="text-anchor:end">concat [h, x],</text>
<text x="280" y="630" class="s rct" style="text-anchor:end">then conv</text>
<rect x="310" y="574" width="180" height="50" rx="6" class="box"/><text x="400" y="598" class="t">recall_conv</text><text x="400" y="616" class="sm">conv 131→128</text>
<path d="M400,574 L400,546" class="ar"/>
<circle cx="400" cy="670" r="24" class="op"/><text x="400" y="676" class="t">cat</text>
<text x="414" y="711" class="s">(B, 131, H, W)</text>
<path d="M400,646 L400,626" class="ar"/>
<!-- loop back -->
<path d="M400,330 L660,330 L660,670 L426,670" class="ar"/>
<text x="676" y="470" class="t" style="text-anchor:start">h(t+1) becomes</text>
<text x="676" y="494" class="t" style="text-anchor:start">the next h(t)</text>
<!-- h0 -->
<circle cx="400" cy="780" r="22" class="op"/><text x="400" y="786" class="t">h₀</text>
<text x="370" y="786" class="s" style="text-anchor:end">(B, 128, H, W)</text>
<path d="M400,758 L400,696" class="ar"/>
<rect x="600" y="758" width="190" height="44" rx="6" class="box" style="stroke-dasharray:6 4"/><text x="695" y="786" class="t">interim_thought</text>
<text x="695" y="824" class="sm">optional</text>
<path d="M598,780 L424,780" class="dash"/><text x="440" y="770" class="s">replaces h₀</text>
<!-- projection & x -->
<rect x="310" y="836" width="180" height="50" rx="6" class="box"/><text x="400" y="860" class="t">projection</text><text x="400" y="878" class="sm">conv 3→128 + ReLU</text>
<path d="M400,836 L400,804" class="ar"/>
<circle cx="400" cy="950" r="22" class="op"/><text x="400" y="956" class="t">x</text>
<text x="432" y="956" class="s">(B, 3, H, W)</text>
<path d="M400,928 L400,888" class="ar"/>
<!-- recall -->
<path d="M378,950 L90,950 L90,670 L374,670" class="rc"/>
<text x="104" y="880" class="s rct">raw x, fed in at</text>
<text x="104" y="900" class="s rct">every iteration (recall)</text>
""")

FIGURES = {"basic_block": BASIC_BLOCK, "feedforward": FEEDFORWARD, "dtnet": DTNET}


def main():
    has_rsvg = shutil.which("rsvg-convert") is not None
    for name, content in FIGURES.items():
        svg_path = OUT_DIR / f"{name}.svg"
        svg_path.write_text(content, encoding="utf-8")
        print(f"wrote {svg_path}")
        if has_rsvg:
            png_path = svg_path.with_suffix(".png")
            subprocess.run(["rsvg-convert", "-z", str(PNG_ZOOM), str(svg_path), "-o", str(png_path)], check=True)
            print(f"wrote {png_path}")
    if not has_rsvg:
        print("rsvg-convert not found: skipped the PNGs")


if __name__ == "__main__":
    main()
