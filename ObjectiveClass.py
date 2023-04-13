

#==============================================================================
class HeriacleObjective():
    '''
     Base Class for designing an objective 
    '''
    #----------------------------------------------------------
    def __init__(self, parentobj=None, nullscore=1.0):
        if not isinstance(nullscore, float):
            raise TypeError("Nullscore must be a floating point value!")
        self.name = ''
        self.parent = parentobj
        self.childlist = []
        self.nullscore = nullscore
    #----------------------------------------------------------
    def __call__(self, depth=0, **kwargs):
        score = 0.0
        score += self.getchildscores(depth=depth+1, **kwargs)
        return score
    #----------------------------------------------------------
    def printinfo(self, depth=0):
        print("Object Type: %s, Depth: %s"%(self.name, depth))
        for child in self.childlist:
            child.printinfo(depth=depth+1)

    #----------------------------------------------------------
    def getnullscores(self, depth=0):
        score = self.nullscore
        print("Depth %s is NULL! Null Score: %s"%(depth, score))
        if len(self.childlist) < 1:
            return score
        for child in self.childlist:
            score += child.getnullscores(depth=depth+1)
        return score
    #----------------------------------------------------------
    def getchildscores(self, depth=0, **kwargs):
        score = 0.0
        if len(self.childlist) < 1:
            return score
        for child in self.childlist:
            score += child(depth=depth+1, **kwargs)
        return score
    #----------------------------------------------------------
    def setparent(self, parentobj):
        self.parent = parentobj
    #----------------------------------------------------------
    def addchild(self, childobj):
        childobj.setparent(self)
        self.childlist.append(childobj)
    #----------------------------------------------------------
#==============================================================================
