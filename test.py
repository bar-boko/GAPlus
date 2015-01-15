__author__ = "Bar Bokovza"

import Code.dataHolder as dat
import Code.compiler as comp

data = dat.GAP_Data()
data.Load("External/Data/fb-net1.csv")
data.Load("External/Data/fb-net2.csv")
data.Load("External/Data/fb-net3.csv")

com = comp.GAP_Compiler()
com.Load("External/Rules/Pi4i.gap")
code = com.Compile()

final_code = ""

for line in code:
    text, tabsCount = line

    command = ""
    for i in range(tabsCount):
        command += "\t"

    command += text
    final_code += command + "\n"

machine_code = compile(final_code, "<string>", "exec")
exec(machine_code)
test = 1

