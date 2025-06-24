import chess
import math
import random
#education attempt for a purely MCTS based chess engine

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.root_player = parent.root_player if parent else state.turn  #for who the root player is\


    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))
    
    def best_child(self, exploration_weight=1.4):
        if not self.children:
            return None

        weights = []
        for child in self.children:
            if child.visits == 0:
                weights.append(float('inf'))  # Always explore unvisited
            else:
                exploitation = child.value / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / child.visits)
                weights.append(exploitation + exploration)

        return self.children[weights.index(max(weights))]
    
    def expand(self):
        legal_moves=list(self.state.legal_moves)
        for move in legal_moves:
            new_state = self.state.copy()
            new_state.push(move)
            self.children.append(Node(new_state, parent=self))

    def rollout(self):
        current_state = self.state.copy()
        while not current_state.is_game_over():
            legal_moves = list(current_state.legal_moves)
            move = random.choice(legal_moves)
            current_state.push(move)
        result=current_state.result()

        if result == '1-0':
            return 1 if self.root_player == chess.WHITE else 0
        elif result == '0-1':
            return 0 if self.root_player == chess.WHITE else 1
        else:
            return 0.5 #    
    
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(1-value)

class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, root):
        for _ in range(self.iterations):
            #print(f"Iteration {_+1}/{self.iterations}")
            node = root
            depth=0
            while node.is_fully_expanded():
                node = node.best_child()
                if node is not None:
                    depth += 1
                    #print(f"visiting depth: ",depth+1 )
            if not node.is_fully_expanded():
                node.expand()
                node = random.choice(node.children)
            value = node.rollout()
            node.backpropagate(value)
        return root.best_child(exploration_weight=0)  # Return the best child without exploration
    

board = chess.Board()
root_node = Node(board)
mcts = MCTS(iterations=1000)
movenum=0
while not board.is_game_over():
    print("move: ", movenum+1)
    best_child = mcts.search(root_node)
    move = best_child.state.move_stack[-1]
    print(f"Engine plays: {move}")
    board.push(move)

    print(board)
    # Let human play (optional)