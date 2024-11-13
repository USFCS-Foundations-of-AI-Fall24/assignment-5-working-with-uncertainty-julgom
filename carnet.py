from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, 
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    #values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent":["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_key_present, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

def main():
    print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

    # Query 1: Given that the car will not move, what is the probability that the battery is not working?
    query1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print("\nQuery 1")
    print("\nP(Battery doesn't work | Car doesn't move:")
    print(query1)

    # Query 2: Given that the radio is not working, what is the probability that the car will not start?
    query2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print("\nQuery 2")
    print("\nP(Car doesn't start | Radio doesn't turn on):")
    print(query2)

    # Query 3: Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    query3_without_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    query3_with_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print("\nQuery 3")
    print("\nP(Radio turns on | Battery works without Gas evidence):")
    print(query3_without_gas)
    print("\nP(Radio turns on | Battery works, Car has Gas):")
    print(query3_with_gas)

    # Query 4: Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas?
    query4_without_gas = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    query4_with_gas = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print("\nQuery 4")
    print("\nP(Ignition fails | Car doesn't move without Gas evidence):")
    print(query4_without_gas)
    print("\nP(Ignition fails | Car doesn't move, Car doesn't have Gas):")
    print(query4_with_gas)

    # Query 5: What is the probability that the car starts if the radio works and it has gas?
    query5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print("\nQuery 5")
    print("\nP(Car starts | Radio works, Car has Gas):")
    print(query5)
    
    # New Query: Probability that the key is not present given that the car does not move
    query_key_not_present = car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"})
    print("\nNew Query")
    print("\nP(Key is not present] | Car doesn't move):")
    print(query_key_not_present)

if __name__ == "__main__":
    main()
