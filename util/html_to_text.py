from html.parser import HTMLParser
import html as html_lib

#TODO: Need fix on HTML parser
class _HTMLToText(HTMLParser):
    BLOCK_TAGS = {
        "p","div","section","article","header","footer","main","aside",
        "ul","ol","li","table","thead","tbody","tfoot","tr","td","th",
        "h1","h2","h3","h4","h5","h6","pre","blockquote","hr","br","dl","dt","dd"
    }
    SKIP_TAGS = {"script","style"}

    def __init__(self):
        super().__init__(convert_charrefs=False)  # we'll unescape ourselves
        self._buf = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self.SKIP_TAGS:
            self._skip += 1
        if t in {"br", "tr", "li", "dt"}:
            self._buf.append("\n")
        elif t in {"p","div","section","article","header","footer","main","aside",
                   "h1","h2","h3","h4","h5","h6","pre","blockquote","ul","ol","dl"}:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self.SKIP_TAGS and self._skip:
            self._skip -= 1
        if t in self.BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data):
        if not self._skip and data:
            self._buf.append(data)

    def handle_entityref(self, name):
        if not self._skip:
            self._buf.append(f"&{name};")

    def handle_charref(self, name):
        if not self._skip:
            self._buf.append(f"&#{name};")

    def get_text(self):
        raw = "".join(self._buf)
        txt = html_lib.unescape(raw)
        # normalize whitespace without over-deleting lines
        lines = [ln.strip() for ln in txt.splitlines()]
        # keep short lines (NIH uses a lot of short label lines)
        return "\n".join(ln for ln in lines if ln)