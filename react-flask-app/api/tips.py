
#cycling var value makes the Combat tips apear in order each time get_tips() is called
def get_tips():
    return tip_switcher(get_var_value())

def get_var_value(filename="varstore.dat"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 1) + 1
        if val > 6:
            val = 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val

def tip_switcher(argument):
    switcher = {

       1:'Misinformation vs Disinformation: Misinformation is false information that is spread, but not necessarily on purpose — think misleading headlines or typos that change the context of stories. Disinformation, on the other hand, is purposely created to mislead and deceive.',
       2:'Fake news is often based at least partially in reality, and weaving truth into the lies is what makes false or misleading stories so appealing.',
       3:'Be aware of where you’re getting your information and ask whether you’re going outside your bubble.',
       4:'To prevent inadvertently spreading false information, hold off on posting any information until you’re sure it’s real.',
       5:'Be able and prepared to critique the news being broadcast, and seek information that is not being broadcast or otherwise prioritized.',
       6:'Propaganda is often associated with material prepared by governments, but activist groups and corporate entities can also engage in propaganda.',

    }
    return switcher.get(argument, "not in switcher range")

def disclaimer():
    return "Misinformation combat tips are from Professor Nicole Cooke's research at the University of South Carolina's College of Information and Communications"


#print(get_tips())

#print(disclaimer())