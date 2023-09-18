f=open("parts_test.inp","r")
Lines=f.readlines()
count=0
found_part=False
found_element=False
found_fiber=False
found_matrix=False
help=False
class Node:
    def __init__(self,coords):
        self.coords=coords
        self.x=coords[0]
        self.y=coords[1]
        self.z=coords[2]
    def get_coords(self):
        return self.x,self.y,self.z
    def __repr__(self):
        return str(x)+", "+str(y)+", "+str(z)
class Element:
    def __init__(self, Nodes):
        self.Nodes=[]
        for n in Nodes:
            self.Nodes.append(n)
    def __repr__(self):
        str=""
        for Node in self.Nodes:
            for i in Node.get_coords():
                str.append(str(i))
                str.append(", ")
            str.append("\n")
        return str
    def get_Nodes(self):
        return self.Nodes
Nodes=[]
Elements=[]
Element=[]
FiberElements=[]
for line in Lines:
    if line:
        l=line.split()
        if l[0]=="*Part,":
            if l[1]=="name=rve":
                found_part=True
                print(line)
        else:
            if found_part:
                if l[0]=="*Element,":
                    found_part=False
                    found_element=True
                    print("hi")
                else:
                    if l[0]!="*Node":
                        x=float(l[1].rstrip(","))
                        y=float(l[2].rstrip(","))
                        z=float(l[3].rstrip(","))
                        Nodes.append(Node([x,y,z]))
                        print(Node([x,y,z]).get_coords())
            else:

                if found_fiber:
                    if l[1]=="elset=Matrix":
                        found_fiber=False
                        found_matrix=True
                    else:
                        for num in l:
                            FiberElements.append(Elements[int(num.rstrip(","))-1])
                if not found_matrix:
                    if l[0]=="*Elset,":
                        found_element=False
                        found_fiber=True
                if found_element:
                    #print("hi")
                    for k in l:
                        #print("hi")
                        if help:
                            Element.append([Nodes[int(k.rstrip(","))-1]])
                            #print("hi")
                        help=True
                    Elements.append(Element)
                    help=False
                    Element=[]


    
    count=count+1
print(Lines[6].split())
print(Elements[203])
print(Nodes)
#print(Elements[1])