

import random
import argparse
import codecs
import os
import numpy

import math

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}
              
             {'#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'}, 
              'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'}, 
              'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'}, 
              'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}}"""


         
        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        emissions_file = basename + ".emit"
    
        with open(emissions_file, "r") as emit_file:
            for line in emit_file:
                parts = line.strip().split() 
                state = parts[0]  
                next_state = parts[1]
                probability = parts[2]

                # Check if the state already exists in transitions
                if state not in self.emissions:
                    self.emissions[state] = {}  # Create a new dictionary if it doesn't exist

                # Add the next_state and its corresponding probability to the state's dictionary
                self.emissions[state][next_state] = probability
        
        transitions_file = basename + ".trans"
    
        with open(transitions_file, "r") as trans_file:
            for line in trans_file:
                parts = line.strip().split() 
                state = parts[0]  
                next_state = parts[1]
                probability = parts[2]

                # Check if the state already exists in transitions
                if state not in self.transitions:
                    self.transitions[state] = {}  # Create a new dictionary if it doesn't exist

                # Add the next_state and its corresponding probability to the state's dictionary
                self.transitions[state][next_state] = probability
   
    ## you do this.
    def generate(self, n):
        # Start from the initial state
        current_state = '#'
        state_sequence = []
        output_sequence = []

        for _ in range(n):
             
            # Choose the next state based on the transition probabilities
            transition_states = list(self.transitions[current_state].keys())
            transition_probs = list(self.transitions[current_state].values())
            
            transition_probs = [float(prob) for prob in transition_probs]

            current_state = random.choices(transition_states, weights=transition_probs)[0]
            state_sequence.append(current_state)
             
            # Choose the output based on the emission probabilities
            emission_states = list(self.emissions[current_state].keys())
            emission_probs = list(self.emissions[current_state].values())
            
            emission_probs = [float(prob) for prob in emission_probs]

            output = random.choices(emission_states, weights=emission_probs)[0]
            output_sequence.append(output)

        return Sequence(state_sequence, output_sequence)
    
    def forward(self, sequence):
        ## you do this: Implement the forward algorithm. Given a Sequence with a list of emissions,
        ## determine the most likely sequence of states.

        sequence.insert(0, '-')
        
        states = list(self.transitions.keys())
        #sequence2 = ['-', 'purr', 'silent', 'silent', 'meow', 'meow']
       
        num_states = len(states)
        num_observations = len(sequence) 
        
        M = numpy.zeros((num_states, num_observations))

        M[states.index('#'), 0] = 1.0
        
        for s in states:  
            if s == '#':
                continue
            if s in self.transitions[states[0]] and sequence[1] in self.emissions[s]:
                M[states.index(s), 1] = float(self.transitions[states[0]][s]) * float(self.emissions[s][sequence[1]]) 
            else:
                M[states.index(s), 1] = 0.0

        for i in range(2, num_observations):
            for s in states:
                if s == '#':
                    continue
                sum_prob = 0
                for s2 in states:
                    if s2 == '#':
                        continue
                    trans_prob = self.transitions[s2][s] if s in self.transitions[s2] else 0.0
                    emit_prob = self.emissions[s][sequence[i]] if sequence[i] in self.emissions[s] else 0.0
                
                    sum_prob += M[states.index(s2), i-1] * float(trans_prob) * float(emit_prob)
                M[states.index(s), i] = sum_prob 
               
        column = M[:, num_observations - 1]
   
        most_likely_state_index = numpy.argmax(column)
        most_likely_state = states[most_likely_state_index]
        
        return most_likely_state

    def viterbi(self, sequence):   
        ## You do this. Given a sequence with a list of emissions, fill in the most likely
        ## hidden states using the Viterbi algorithm.
        sequence.insert(0, '-')
        states = list(self.transitions.keys())
        #sequence2 = ['-', 'purr', 'silent', 'silent', 'meow', 'meow']

        num_states = len(states)
        num_observations = len(sequence) 
        
        M = numpy.zeros((num_states, num_observations))
        B = numpy.zeros((num_states, num_observations))

        M[states.index('#'), 0] = 1.0
        
        for s in states:  
            if s == '#':
                continue
            if s in self.transitions[states[0]] and sequence[1] in self.emissions[s]:
                M[states.index(s), 1] = float(self.transitions[states[0]][s]) * float(self.emissions[s][sequence[1]]) 
            else:
                M[states.index(s), 1] = 0.0

        for i in range(2, num_observations):
            for s in states:
                if s == '#':
                    continue
                max_val = float('-inf')  
                best_state = None
                for s2 in states:
                    if s2 == '#':
                        continue
                    trans_prob = self.transitions[s2][s] if s in self.transitions[s2] else 0.0
                    emit_prob = self.emissions[s][sequence[i]] if sequence[i] in self.emissions[s] else 0.0
                
                    val = M[states.index(s2), i-1] * float(trans_prob) * float(emit_prob)
                    
                    if val > max_val:  
                        max_val = val
                        best_state = s2  
                
                M[states.index(s), i] = max_val
                B[states.index(s), i] = states.index(best_state)  

        best_path = []
        best_index = numpy.argmax(M[:, num_observations - 1])  
        best = states[best_index]
        best_path.append(best)  

        for i in range(num_observations - 1, 1, -1):
            best_index = int(B[best_index, i]) 
            best = states[best_index]
            best_path.append(best)
            
        best_path.reverse() 
        return best_path

       
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate random sequences using HMM.")
    parser.add_argument("model", help="Base name of the HMM model files (without extension)")
    parser.add_argument("--generate", type=int, help="Number of observations to generate")
    parser.add_argument("--forward", type=str, help="Observation file for running the forward algorithm")
    parser.add_argument("--viterbi", type=str, help="Observation file for running the viterbi algorithm")

    args = parser.parse_args()
    
    h = HMM()
    h.load(args.model)
    
    if args.generate:
        sequence = h.generate(args.generate)
        print(sequence)

        file_name = f"{args.model}_sequence.obs"
        
        with open(file_name, "w") as file:
            file.write(" ".join(sequence.outputseq))
        
    if args.forward:
        with open(args.forward, 'r') as obs_file:
            observations = obs_file.read().strip().split()
            most_likely_state = h.forward(observations)
            if args.model == 'lander':
                if most_likely_state in ['2,5', '3,4', '4,3', '4,4', '5,5']:
                    most_likely_state = "safe to land"
                else:
                    most_likely_state = "not safe to land"
            print(most_likely_state)
        
    if args.viterbi:
        with open(args.viterbi, 'r') as obs_file:
            observations = obs_file.read().strip().split()
            best_path = h.viterbi(observations)
            print(best_path)