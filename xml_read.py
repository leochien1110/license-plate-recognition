# ref: https://officeguide.cc/python-read-write-xml-format-file-tutorial-examples/

from xml.etree import ElementTree as ET

# load file and parse xml structure
tree = ET.parse("data/train/annotations/Cars0.xml")
root = tree.getroot()

# check tags
for child in root:
    print(child.tag, child.text)

# find object's names
for child in root.findall('object'):
    name = child.find('name')
    print(name.text)

# modify xml and save as new file
for child in root.findall('object'):
    name = child.find('name')
    name.text = "license"
    tree.write("Cars0_us.xml")