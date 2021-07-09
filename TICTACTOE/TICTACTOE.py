empty=None
import copy
import random
X='X'
O='O'

# INITIAL STATE OF THE BOARD
def initial():

    return [[empty, empty, empty],
            [empty, empty, empty],
            [empty, empty, empty]]

# RETURNS ALL POSSIBLE CHOICES FOE THE BOARD
def action(state):
    chance = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == empty:
                chance.append((i, j))
    return chance


#CHECKS IF THE GAME HAS REACHED DRAW
def terminal(state):
    elements=[]
    for i in range(3):
        for j in range(3):
            elements.append(state[i][j])
    if elements.count(empty) > 0:
        return False
    else:
        return True

#CHECKS IF A GAME HAS A WINNER
def utility(state):
    elements = []
    for i in range(3):
        for j in range(3):
            elements.append(state[i][j])
    for value in [X,O]:
        if value==X:
            q=1
        else:
            q = -1
        for i in state:
            if i == [value, value, value]:
                return q
        if elements[0] == value and elements[3] == value and elements[6] == value:
            return q
        elif elements[1] == value and elements[4] == value and elements[7] == value:
            return q
        elif elements[2] == value and elements[5] == value and elements[8] == value:
            return q
        elif elements[0] == value and elements[4] == value and elements[8] == value:
            return q
        elif elements[2] == value and elements[4] == value and elements[6] == value:
            return q
    return 0

#PART 1 OF MINIMAX ALGORITHM
def maxvalue(state):
    if terminal(state):
        if utility(state) == 0:
            return 0
        else:
            return utility(state)
    elif utility(state) == 1:
        return 1
    elif utility(state)==-1:
        return -1
    v=-float('inf')
    for i in action(state):
        v = max(v, minvalue(result(state, i, X)))
    return  v

#PART 2 MINIMAX ALGORITHM
def minvalue(state):
    if terminal(state):
        if utility(state) == 0:
            return 0
        else:
            return utility(state)
    elif utility(state) == 1:
        return 1
    elif utility(state) == -1:
        return -1
    v=float('inf')
    for i in action(state):
        v=min(v,maxvalue(result(state,i,O)))
    return v

#GIVEN A BOARD AND ACTION ,RETURNS THE BOARD WITH ACTION COMPLETED
def result(state, action, val):
    import copy
    q = copy.deepcopy(state)

    q[action[0]][action[1]] = val
    return q

#SORTS A TUPLE
def Sort_Tuple(tup):
    tup.sort(key=lambda x: x[1])
    return tup


#GIVES THE BEST OPTIMAL MOVE FOR THE CURRENT STATE OF THE GAME
def funct(state):
    import random
    import copy
    l=[]
    q = copy.deepcopy(state)
    val=O
    for i in action(q):
        l.append(result(q,i,val))
    o=[]
    w=[]
    for  i in l:
        o.append((i,maxvalue(i)))
    Sort_Tuple(o)
    k2 = []
    for i,j in o:
        if j==o[0][1]:
            k2.append(i)

    return random.choice(k2)

#PRINTS THE BOARD
def printer(matrix):
    for i in range(3):
        for j in range(3):
            if matrix[i][j]==X or matrix[i][j]==O:
                print(matrix[i][j], end = "    ")
            else:
                print(matrix[i][j],end=' ')
        print()

#dictionaries mapping input and action
positions={'7':(0,0),
           '8': (0,1),
           '9':(0,2),'4':(1,0),'5':(1,1),'6':(1,2),'1':(2,0),'2':(2,1),'3':(2,2)
           }
positions_reverse={(0,0):'7',
           (0,1):'8',
           (0,2):'9',(1,0):'4',(1,1):'5',(1,2):'6',(2,0):'1',(2,1):'2',(2,2):'3'}


# TO TAKE YOUR CHOICES
moves=[]
while True:
    print()
    choice=input(' WHAT IS YOUR CHOICE X OR O  ').capitalize()
    print()
    order=input(' WANNA GO 1st OR 2nd ')
    if choice in ['X','O'] and order  in ['1','2']:
        break
    else:
        print()
        print(' INVALID CHOICE')

print()


if order=='1':
    new=initial()
else:
    new=initial()
    i=random.randint(0,2)
    l=random.randint(0,2)
    new[i][l]=O
    moves.append(positions_reverse[(i,l)])
printer(new)
print()
# THE GAME
while True:
    if new == initial():
        pass
    elif utility(new) == 1:
        print(' YOU WIN')
        break
    elif utility(new) == -1:
        print(' YOU LOSE ')
        break
    while True:
        on = input(' WHAT IS YOUR MOVE  ')
        print()
        if len(on)==1:
            if on in list(positions.keys()):
                if on in moves:
                    print(' MOVE ALREADY USED')
                    print()
                else:
                    moves.append(on)
                    actiom=positions[on]
                    break
            else:
                print(' INVALID MOVE')
        elif len(on)>1 :
            print(' INVALID MOVE')
            print()
    new = result(new, actiom, X)
    if terminal(new):
        if utility(new) == 0:
            print(' ITS A DRAW')
            break
    new = funct(new)
    for i in range(3):
        for j in range(3):
            if new[i][j]==O:
                if positions_reverse[(i,j)] not in moves:
                    moves.append(positions_reverse[(i,j)])
    if choice=='O':
        q = copy.deepcopy(new)
        for i in range(3):
            for j in range(3):
                if q[i][j]==X:
                    q[i][j]=O
                elif q[i][j]==O:
                    q[i][j]=X
        printer(q)
    else:
        printer(new)
    print()




