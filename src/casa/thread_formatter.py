from io import StringIO
from textwrap import indent, wrap
from datetime import datetime

def format_text(text, textwidth=60, __indent=2, __indent0=0):
    return "\n".join(
        indent(ln, ' '*(__indent0 if ln_i==0 else __indent))
        for ln_i, ln in enumerate(wrap(text, textwidth))
    )

def format_thread(thread):
    sio = StringIO()
    timestamp = datetime.strptime(t0.main.id[1:15], 
                                  "%Y%m%d%H%M%S").strftime("%c")
    sio.write(f"[{thread.source}] ")
    sio.write(timestamp + "\n")
    if thread.main:
        sio.write(format_text(thread.main.title, __indent=0))
        sio.write("\n")
        sio.write(format_text("----", __indent=0))
        sio.write("\n")
        sio.write(format_text(thread.main.text, __indent=0))
        sio.write("\n\n")
    for reply_x in thread.replies:
        sio.write(format_text(">> " + reply_x.text, __indent=3, __indent0=0))
        sio.write("\n")
    return sio.getvalue()

def format_thread_html(thread, idx=-1):
    sio = StringIO()
    op0 = thread.get_opinion()
    if not op0.id:
        timestamp = ""
    else:
        timestamp = datetime.strptime(op0.id[1:15], 
                                  "%Y%m%d%H%M%S").strftime("%c")
    sio.write(f"<div class='thread-wrapper' id='{op0.id}''>\n")
    sio.write(f"<span class='source'>{idx}. [{thread.source}] </span>")
    sio.write("<span class='timestamp'>" + timestamp + "</span>\n")
    if thread.main:
        sio.write("<div class='title'>")
        sio.write(format_text(thread.main.title, __indent=0))
        sio.write("</div>\n<div class='main-text'>\n")
        sio.write(format_text(thread.main.text, __indent=0))
        sio.write("</div>\n\n")
    else:
        sio.write("<div class='title'>(無主文)</div>")
    sio.write("<ol class='reply-wrap'>\n")
    for reply_idx, reply_x in enumerate(thread.replies):
        sio.write(format_text(f"  <li class='reply-text' id='reply-{reply_idx}'> " 
            + reply_x.text, __indent=3, __indent0=0))
        sio.write("  </li>\n")
    sio.write("</ol>\n")
    sio.write("</div>")
    return sio.getvalue()

def format_thread_by_type_html(thread, idx=-1, optype="main"):
    sio = StringIO()
    op0 = thread.get_opinion()

    sio.write(f"<div class='thread-part-wrapper' id='{op0.id}-{optype}'>\n")
    if optype == "main":
        if not op0.id:
            timestamp = ""
        else:
            timestamp = datetime.strptime(op0.id[1:15], 
                                    "%Y%m%d%H%M%S").strftime("%c")    
        sio.write(f"<span class='source'>{idx}. [{thread.source}] </span>")
        sio.write("<span class='timestamp'>" + timestamp + "</span>\n")
        if thread.main:
            sio.write("<div class='title'>")
            sio.write(format_text(thread.main.title, __indent=0))
            sio.write("</div>\n<div class='main-text'>\n")
            sio.write(format_text(thread.main.text, __indent=0))
            sio.write("</div>\n\n")
        else:
            sio.write("<div class='title'>(無主文)</div>")
    
    elif optype == "reply":        
        sio.write("<ol class='reply-wrap'>\n")
        for reply_idx, reply_x in enumerate(thread.replies):
            sio.write(format_text(f"  <li class='reply-text' id='reply-{reply_idx}'> " 
                + reply_x.text, __indent=3, __indent0=0))
            sio.write("  </li>\n")
        sio.write("</ol>\n")
    
    else:
        raise ValueError("not supported opType: " + optype)
    sio.write("</div>")
    return sio.getvalue()