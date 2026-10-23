import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

root=Path(__file__).resolve().parents[1]
src=root/'DataManagement/AswendtLab_Data_Management_Plan_v1.md'
out=root/'output/pdf/AswendtLab_Data_Management_Plan_v1.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
styles=getSampleStyleSheet()
for name,size,leading,space in [('BodyText',10,14,7),('Heading1',23,27,14),('Heading2',16,20,11),('Heading3',11.5,15,7)]:
    s=styles[name]; s.fontName='Helvetica-Bold' if name!='BodyText' else 'Helvetica'; s.fontSize=size; s.leading=leading; s.spaceAfter=space
    if name!='BodyText': s.textColor=colors.HexColor('#173b50'); s.keepWithNext=True
styles.add(ParagraphStyle(name='CaptionDMP',parent=styles['BodyText'],fontSize=8.8,leading=12,textColor=colors.HexColor('#546773')))
styles.add(ParagraphStyle(name='BulletDMP',parent=styles['BodyText'],leftIndent=12,firstLineIndent=-8))
styles.add(ParagraphStyle(name='TableDMP',parent=styles['BodyText'],fontSize=8.6,leading=11,spaceAfter=0))
def fmt(t):
    t=escape(t)
    t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
    t=re.sub(r'`(.+?)`',r'<font face="Courier" size="8.5">\1</font>',t)
    def link(m):
        label, target = m.groups()
        if not target.startswith(('https://', 'http://')):
            relative = (src.parent / target).resolve().relative_to(root).as_posix()
            target = 'https://github.com/Aswendt-Lab/Tools/blob/main/' + relative
        return '<a href="' + target + '" color="#087f83"><u>' + label + '</u></a>'
    t=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,t)
    return t
story=[]
lines=src.read_text().splitlines(); i=0
while i<len(lines):
    line=lines[i].strip(); i+=1
    if not line: continue
    if line=='---': story.append(PageBreak()); continue
    if line.startswith('!['):
        match=re.match(r'!\[.*?\]\((.*?)\)',line)
        img=Image(str(src.parent/match.group(1)))
        factor=min(495/img.imageWidth,530/img.imageHeight)
        img.drawWidth=img.imageWidth*factor; img.drawHeight=img.imageHeight*factor
        img.hAlign='LEFT'
        story.extend([img,Spacer(1,7)])
        continue
    if line.startswith('*') and line.endswith('*') and not line.startswith('**'):
        story.append(Paragraph('<i>'+fmt(line[1:-1])+'</i>',styles['CaptionDMP']))
        continue
    if line.startswith('|'):
        rows=[line]
        while i<len(lines) and lines[i].startswith('|'): rows.append(lines[i]); i+=1
        data=[]
        for row in rows:
            cells=[c.strip() for c in row.strip('|').split('|')]
            if all(re.fullmatch(r'[-:]+',c) for c in cells): continue
            data.append([Paragraph(fmt(c),styles['TableDMP']) for c in cells])
        t=Table(data,colWidths=[140,165,190],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef2')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#bccbd2')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([t,Spacer(1,10)]); continue
    if line.startswith('# '): style='Heading1'; line=line[2:]
    elif line.startswith('## '): style='Heading2'; line=line[3:]
    elif line.startswith('### '): style='Heading3'; line=line[4:]
    elif line.startswith('- '): style='BulletDMP'; line='&#8226; '+fmt(line[2:]); story.append(Paragraph(line,styles[style])); continue
    else: style='BodyText'
    story.append(Paragraph(fmt(line),styles[style]))
def furniture(c,d):
    c.saveState()
    c.setFillColor(colors.HexColor('#546773'))
    c.setStrokeColor(colors.HexColor('#b8c9d2')); c.line(50,801,545,801)
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#546773'))
    c.drawString(50,812,'ASWENDT LAB  /  RESEARCH DATA MANAGEMENT')
    c.drawString(50,28,'Version 1.0 | 23 October 2026 | Draft for lab adoption')
    c.drawRightString(545,28,str(d.page))
    c.restoreState()
doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=58,bottomMargin=48,title='Aswendt Lab Data Management Plan',author='Aswendt Lab')
doc.build(story,onFirstPage=furniture,onLaterPages=furniture)
print(out)
