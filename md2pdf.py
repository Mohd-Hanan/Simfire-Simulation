from markdown_pdf import MarkdownPdf, Section
import os

pdf = MarkdownPdf(toc_level=2)
md_path = "/home/hari/PycharmProjects/Simfire-Simulation/project_summary.md"
with open(md_path, "r") as f:
    pdf.add_section(Section(f.read()))

pdf.save("/home/hari/PycharmProjects/Simfire-Simulation/Project_Summary_SimFire.pdf")
print("PDF created.")
