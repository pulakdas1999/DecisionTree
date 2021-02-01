training_data = [
    ['Green',3,'Mango'],
    ['Yellow',3,'Mango'],
    ['Red',1,'Grape'],
    ['Red',1,'Grape'],
    ['Yellow',3,'Lemon']
]
header = ['Color','Diameter','Label']
# Find the unique values for a column in dataset.
def unique_vals(rows,columns):
    return set([rows[columns] for row in rows])
# Counts the number of each type of example in adataset.
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label]=0
        counts[label]+=1
    return counts
# Test if a value is numeric.
def isnumeric(values):
    return isinstance(values,int) or isinstance(values,float)
# A question is used to partition a dataset.
# This class just records a column number(e.g., 0 for colour) and a column value("Green").The match method is used to compare the
# feature value in an e.g. to the feature value stored in the question.
class Question:
    def __init__(self,column,value):
        self.column = column
        self.value = value
    def match(self,example):
        # Compare the feature value in an e.g. to the feature value in in this question.
        val = example[self.column]
        if isnumeric(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        # This is just a helper method to print the question in a readable format.
        condition = "=="
        if isnumeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column],condition,str(self.value))
# Partitions is a dataset.For each row in dataset check if it matches the question.If so add it to true_rows else to false_rows.
def partition(rows,question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows,false_rows
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl]/float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
def information_gain(left,right,current_uncertainty):
    p = float(len(left))/(len(left)+len(right))
    return current_uncertainty-p*gini(left)-(1-p)*gini(right)
def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0])-1
    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col,val)
            true_rows,false_rows=partition(rows,question)
            if len(true_rows)==0 or len(false_rows)==0:
                continue
            gain = information_gain(true_rows,false_rows,current_uncertainty)
            if gain >= best_gain:
                best_gain,best_question = gain,question
    return best_gain,best_question
class leaf:
    def __init__(self,rows):
        self.predictions = class_counts(rows)
class decision_Node:
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
def build_tree(rows):
    gain,question = find_best_split(rows)
    if gain==0:
        return leaf(rows)
    true_rows,false_rows = partition(rows,question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return decision_Node(question,true_branch,false_branch)
def print_tree(node,spacing=""):
    if isinstance(node,leaf):
        print(spacing +"predict",node.predictions)
        return
    print(spacing+str(node.question))
    print(spacing+'---> TRUE:')
    print_tree(node.true_branch,spacing+" ")
    print(spacing+'---> FALSE:')
    print_tree(node.false_branch,spacing+" ")
def classify(row,node):
    if isinstance(node,leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row,node.true_branch)
    else:
        return classify(row,node.false_branch)
def print_leaf(counts):
    total = sum(counts.values())*1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl]=str(int(counts[lbl]/total * 100))+"%"
    return probs
if __name__=='__main__':
    my_tree = build_tree(training_data)
    print_tree(my_tree)
    testing_data = [
        ['Green', 3, 'Mango'],
        ['Yellow', 4, 'Mango'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
    ]
    for row in testing_data:
        print("Actual : %s. Predicted : %s"%(row[-1],print_leaf(classify(row,my_tree))))
        