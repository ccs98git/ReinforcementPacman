# Fantasma_Agents.py
# vbase 1.02 22/04/19

import random, sys
import util, os
import math
import time, datetime
from shutil import copyfile
from game import Agent
from game import Directions
import inference
import busters
from game import Actions
from game import Directions
from keyboardAgents import KeyboardAgent
from bustersAgents import BustersAgent
from distanceCalculator import Distancer
from learningAgents import ReinforcementAgent

# CONSTANTS

#  CREATORS
#  NOTA: el id debe ser numerico sin el 100 delante
#  NOTA: nombre en formato "nombre apellidos"
CREATORS = {
    363633: 'Claudia Castelo Sagnotti',
    363565: 'Victor Gomez de la Camara'
}

# Flags
TRACE_SHOW_QTABLE = True
TRACE_ACTION_DECISION = True
TRACE_UPDATE = True
TRACE_BEST_ACTION = True
TRACE_STATE = True


class Fantasma_Agents(BustersAgent,ReinforcementAgent):
    """
        UC3M - Campus de Colmenarejo
        Aprendizaje Automatico - 2019
        Practica 2

        Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

        Seguro que tendras que cambiar...
            CREATORS (constante global)
            def createInitialQTable(self):
            def computeQLearningState(self, gameState):
            def getPossibleActions(self, state):
            def update(self, state, action, nextState, reward):

        Puede que quieras cambiar...
            def getActionQLearning(self, state):
            def chooseAction(self, gameState):
            def __init__(self, **args):
            def readQtable(self):
            def writeQtable(self):
            def getActionIndex(self, action):
            def getQValue(self, state, action):
            def setQValue(self, state, action, value):
            def computeValueFromQValues(self, state):
            def computeActionFromQValues(self, state):
            def getActionQLearning(self, state):
            def getPolicy(self, state):
            def getValue(self, state):
            def final(self, state):

        Seguramente no necesites cambiar...
            def trace(self, traceString):
            def getAgentName(self):
            def getQTableFileName(self):
            def registerInitialState(self, gameState):

            Ejemplo de parametros:
            -n 1000 -k 1 -l labAA1.lay -p UC3M_AA_Prac2_Example -a "alpha=0.8,epsilon=0.2,gamma=0.8" -m 2000

    """
    def __init__(self, **args):
        "Initialize Q-values"

        # Separamos los argumentos de la superclase BusterAgent
        bustersArgs={}
        bustersArgs['ghostAgents']=args['ghostAgents']

        # y los de la clase ReinforcementAgent
        reinforcementArgs={}
        for i in ('alpha','epsilon','gamma'):
            reinforcementArgs[i]=args[i]
        reinforcementArgs['actionFn'] = self.getPossibleActions

        BustersAgent.__init__(self,**bustersArgs)
        ReinforcementAgent.__init__(self,**reinforcementArgs)

        # Traza comun de arranque en todos los agentes
        self.trace('Init::AgentName='+self.getAgentName())
        for creator in enumerate(CREATORS):
            self.trace('Init::Creator '+str(creator[0]+1)+' ' + str(creator[1]) + '>' + CREATORS[creator[1]])
        self.trace('Init::Q-Learning initialization values')
        for i in ('alpha','epsilon','gamma'):
            self.trace('      ' + str(i) + '=' + str(args[i]))

        if not os.path.exists(self.getQTableFileName()):
            self.createInitialQTable()
        self.table_file = open(self.getQTableFileName(), "r+")

        # Try to read and then to write to detect problems at the beginning
        self.q_table = self.readQtable()
        self.writeQtable()

        # Statictics initialization
        self.stats = {
            'ticks':0,
            'finalScore':0,
            'QTableUpdates':0
        }



    ########################################################################################
    ########################################################################################
    ## Parte de Gestion del Agente
    ##
    ########################################################################################
    ########################################################################################

    def getAgentName(self):
        """
        Returns the Agent Name (same name as class name)
        :return: string
        """
        return self.__class__.__name__

    def getQTableFileName(self):
        """
        Returns the default qtable file name based on agent name
        :return: string
        """
        return "qtable_"+self.getAgentName()+".txt"

    def createInitialQTable(self):
        """
        Creates a initial Q Table
        :return: nothing
        HINT: Create the minimum lines to allow readQTable not to crash the first time
        """

        f = open(self.getQTableFileName(), "w")
        f.writelines("A 0.0 0.0 0.0 0.0 0.0")
        f.flush()
        f.close()

    def trace(self, traceString):
        """
        Prints a trace with a timestamp
        :param traceString:
        :return:
        """

        ts=time.time()
        tt=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print ("*AATRC*<"+tt+">"+str(traceString))


    ########################################################################################
    ########################################################################################
    ## Parte Q-LEARNING del Agente
    ##
    ########################################################################################
    ########################################################################################

    def getPossibleActions(self, state):
        """
        Devuelve las posibles acciones a tomar por q-learning en un q-estado state
        :param state: string con un q-estado
        :return: lista de strings con las posibles direcciones ['North','South'...]
        """

        return [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]

    def readQtable(self):
        "Read qtable from disc"

        # HINT: En esta funcion hay que implementar como se lee el fichero donde se almacena la tabla Q
        table = self.table_file.readlines()
        q_table = {}

        for i, line in enumerate(table):
            row = line.split()
            state_key = row[0]
            actions = []
            for n in range(1,len(row)):
                actions.append(float(row[n]))
            q_table[state_key]=actions

        return q_table

    def writeQtable(self):
        "Write qtable to disc"

        # HINT: En esta funcion hay que implementar como se escribe el fichero donde se almacena la tabla Q

        # Si alpha es cero, no vamos a tocar la tabla Q
        if self.alpha == 0: return

        self.table_file.seek(0)
        self.table_file.truncate()

        if TRACE_SHOW_QTABLE: self.trace("\n\nQTABLE>\n")

        for i,(k,v) in enumerate(sorted(self.q_table.items())):
            vstr = ' '.join(['{:.4f}'.format(x) for x in v])
            self.table_file.write(k + ' ' + vstr + '\n')

            if TRACE_SHOW_QTABLE: self.trace("  ["+k+"]="+ '(' + vstr + ')')
        self.table_file.flush()


    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()


    def computeQLearningState(self, gameState):
        """
        Calcula el q-estado a partir de un gameState de PacMan
        :param gameState: gameState de PacMan
        :return: un String que sera la clave para acceder a la tabla q
        """
        key_state = ""
	fantasmas_vivos = gameState.getLivingGhosts()
	auxD = gameState.data.ghostDistances
        for i in range (len(gameState.getLegalPacmanActions())):
            key_state += str(gameState.getLegalPacmanActions()[i])
        for i in range(len(gameState.data.ghostDistances)):
	    if(fantasmas_vivos[i+1]==True):
		if (auxD[i] <= 5):
                    key_state= "Muy_Cerca"
                    cGhost = i
		elif (auxD[i] > 5) and (auxD[i] <= 10):
                    key_state += "Cerca"
                    cGhost = i
		elif (auxD[i] > 10) and (auxD[i] <= 15):
                    key_state += "Lejos"
                    cGhost = i
		elif (auxD[i] > 15):
                    key_state += "Muy_Lejos"
                    cGhost = i
		ghost = gameState.getGhostPositions()
                nGhost = ghost[cGhost]
                pacman = gameState.getPacmanPosition()
                x1 = nGhost[0]
                y1 = nGhost[1]
                x2 = pacman[0]
                y2 = pacman[1]
                if(x1>=x2): key_state += "Este"
                elif(x1<x2): key_state += "Oeste"
                elif(y1>=y2): key_state += "Norte"
                elif(y1<y2): key_state += "Sur"
                elif(x1==x2) and (y1<y2): key_state += "Sur"
                elif(x1==x2) and (y1>y2): key_state += "Norte"
                elif(y1==y2) and (x1<x2): key_state += "Oeste"
                elif(y1==y2) and (x1>x2): key_state += "Este"
        if(gameState.getDistanceNearestFood()<=5)and(gameState.getDistanceNearestFood()>0):key_state += "Muy_Cerca"
        elif(gameState.getDistanceNearestFood()>5) and (gameState.getDistanceNearestFood()<=10):key_state += "Cerca"
        elif(gameState.getDistanceNearestFood()>10) and (gameState.getDistanceNearestFood()<=15):key_state += "Lejos"
        elif(gameState.getDistanceNearestFood()>15):key_state += "Muy_Lejos"
        # HINT: Calcular el q-estado a partir del gameState es fundamental
        return key_state

    def getActionIndex(self, action):
        """
            get an action like 'North' and returns an integer
            :param action: 'North' 'South'...
            :return: 0 for 'North', 1 for 'South'..
        """
        if   action == Directions.NORTH: return 0
        elif action == Directions.SOUTH: return 1
        elif action == Directions.EAST: return 2
        elif action == Directions.WEST: return 3
        return 4 # should be STOP

    def getQValue(self, state, action):
        """
            Returns Q(state,action)
            Access the q-table and return the q-value for a given state and action
            Should return 0.0 if we have never seen a state
        :param state: a string q-state
        :param action: a string with an action 'North', 'South'...
        :return: float with the value
        """

        # if not in table, init to 0
        if state not in self.q_table:
            self.q_table[state]=[float(0), float(0), float(0), float(0), float(0)]

        actionIndex = self.getActionIndex(action)
        qtable_row = self.q_table[state]
        q_value = qtable_row[actionIndex]

        return q_value

    def setQValue(self, state, action, value):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        :param state: string con el q-estado
        :param action: string con una accion 'North' 'South'
        :param value: float con el valor de Q(state,action)
        """
        returnValue = float(0)
        if state in self.q_table:
            returnValue = value

        actionIndex = self.getActionIndex(action)
        self.q_table[state][actionIndex] = value

        return returnValue

    def computeValueFromQValues(self, state):
        """
        V(state)
        Calcula el maximo valor de Q para todas las posibles acciones de state
        :param state: string with q-state
        :return: float con V(state)
        """

        legalActions = self.getLegalActions(state)

        if len(legalActions) == 0:
            return 0

        if state not in self.q_table:
            return 0

        maxValue = max(self.q_table[state])
        return maxValue


    def computeActionFromQValues(self, state):
        """
        pi(state)
        Calcula la accion asociada al maximo valor de Q para todas las posibles acciones de state
        NOTA: las acciones legales del q-learning no tienen porque ser mismas las acciones legales de PacMan
        :param state: string with q-state
        :return: string con accion 'North' 'South'
        """

        # legalActions for q-learning (not pacman)
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions[1:]:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        if TRACE_BEST_ACTION: self.trace("["+state+"] best actions="+str(best_actions)+" v="+str(best_value))

        return random.choice(best_actions)


    def getActionQLearning(self, state):
        """
        Calcula la accion a tomar utilizando Epsilon Greedy
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.
        :param state: string with q-state
        :return: string con accion 'North' 'South'
        """

        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            action = random.choice(legalActions)
            if TRACE_ACTION_DECISION: self.trace("Random Action = "+action)
        else:
            action = self.getPolicy(state)
            if TRACE_ACTION_DECISION: self.trace("Policy Action = "+action)

        return action


    def update(self, state, action, nextState, reward):
        """
        Actualiza la q-table con los parametros indicados
        :param state: string con el PacMan-State en t=-1
        :param action: string con la accion en t=-1 'North', 'South'
        :param nextState: string con el PacMan-State en t=0
        :param reward: float con el refuerzo de la transicion
        :return: None

        HINTS:

        if terminal_state:
            Q(state, action) < - (1 - self.alpha)
            Q(state, action) + self.alpha * (r + 0)
        else:
            Q(state, action) < - (1 - self.alpha)
            Q(state, action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

            "*** YOUR CODE HERE ***"
        """
        """
        estado = self.computeQLearningState(state)
        sig = self.computeQLearningState(nextState)
        valorQ = self.getQValue(estado,action)
        valor1 = self.getValue(sig)
        valor1_descontado = valor1 * self.discount
        delta = reward + valor1_descontado
        delta = delta - valorQ
        delta = self.alpha * delta
        self.setQValue(estado, action, valorQ + delta)
        """

        q_state = self.computeQLearningState(state)
        q_nextState = self.computeQLearningState(nextState,)

        Q_St_At = self.getQValue(q_state, action)
        V_St1 = self.getValue(q_nextState)
        V_St1_x_discount = V_St1 * self.discount
        delta = reward + V_St1_x_discount
        delta = delta - Q_St_At
        delta = self.alpha * delta

        self.setQValue(q_state, action, Q_St_At + delta)
        # Reemplazar las asignaciones siguientes por su valor adecuado
        q_state = ""
        q_nextState = ""

        if TRACE_STATE: self.trace("Update State ["+ str(q_state) +"]")
        if TRACE_UPDATE: self.trace("Update State [" + str(q_state) + "][" + action + "]=>[" + str(q_nextState) + "] r=" + str(reward))

        # MUST! Actualizar la cantidad de actualizaciones a la QTable
        #    sera necesario distinguir cuando las actualizaciones son necesarias o no
        self.stats['QTableUpdates'] += 1



    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)


    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def final(self, state):
        """
        Shows episode statistics and call superclass final
        :param state:
        :return:
        """

        self.stats['finalScore'] = state.getScore()

        self.trace('Final Statictics:')
        for stat in enumerate(self.stats):
            self.trace('  '+str(stat[0])+"/"+str(stat[1])+"="+str(self.stats[stat[1]]))

        return ReinforcementAgent.final(self, state)


    ########################################################################################
    ########################################################################################
    ## Parte PACMAN del Agente
    ##
    ########################################################################################
    ########################################################################################
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0


    def chooseAction(self, gameState):
        """
        Funcion que es llamada por game.py para que PacMan tome una direccion
        NOTA: Nunca se deben devolver acciones ilegales para evitar excepciones
        :param gameState: gameState de PacMan
        :return: Direction.*
        """

        #######################################
        ### MAY CHANGE THIS PART -- START
        #
        # calculate the current q-state from the current pacman-gameState
        key_state = self.computeQLearningState(gameState)

        # calculate the action with q-learning algorithm
        action = self.getActionQLearning(key_state)
        #
        ### MAY CHANGE THIS PART -- STOP
        ##########################################

        # MUST! Count the ticks
        self.stats['ticks'] += 1

        # MUST! Notify the superclass the action we have just done
        ReinforcementAgent.doAction(self, gameState, action)

        # MUST! check legal, if not legal STOP to force negative reward
        legal = gameState.getLegalPacmanActions()
        if action not in legal:
            return Directions.STOP

        # MUST! Return the action to be taken to Pacman game
        return action

