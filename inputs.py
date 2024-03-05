run = 1
materials = ["low","high","mid"]
layerStack = []
while run == 1:
    currentLayer= []
    material = (str(input("Select material: "))).lower()
    if material in materials:
        try:
            thickness = float(input("Input the thickness of the material: "))
        except:
            print("Thickness not accepted. Layer not added. ")
        else:
            currentLayer= [material , thickness]
            layerStack.append(currentLayer)
            print(layerStack)
        
    else:
        print("Material not found. Layer not added.")
        print(layerStack)
        
    a = input("Do you wish to select another material? (y/n) ")
    if a != "y":
        run = 0
        print("Program started with chosen settings.")
    