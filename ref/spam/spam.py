from pathlib import Path
import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq

data = Path("SMSSpamCollection").read_text()
data = data.strip()
data = data.split("\n")

# initialize an empty vector
digit_counts = np.empty((len(data), 2), dtype=int)

# split label and mesage
for i, line in enumerate(data):
    case, message = line.split("\t")
    num_digits = sum(c.isdigit() for c in message)
    digit_counts[i, 0] = 0 if case == "ham" else 1
    digit_counts[i, 1] = num_digits

unique_counts = np.unique(digit_counts[:, 1], return_counts=True)
unique_counts = np.transpose(np.vstack(unique_counts))
print(unique_counts)
import sys
sys.exit()


# normalize a group of observations on a per feature basis before applying kmeans
whitened_counts = whiten(unique_counts)
codebook, _ = kmeans(whitened_counts, 3)
# assign codes from a code book to observations
codes, _ = vq(whitened_counts, codebook)
ham_code = codes[0]
spam_code = codes[-1]
unknown_code = list(set(range(3)) ^ set((ham_code, spam_code)))[0]

print("definitely ham:", unique_counts[codes == ham_code][-1])
print("definitely spam:", unique_counts[codes == spam_code][-1])
print("unknown:", unique_counts[codes == unknown_code][-1])

digits = digit_counts[:, 1]
predicted_hams = digits == 0
predicted_spams = digits > 20
predicted_unknowns = np.logical_and(digits > 0, digits <= 20)

spam_cluster = digit_counts[predicted_spams]
ham_cluster = digit_counts[predicted_hams]
unk_cluster = digit_counts[predicted_unknowns]

print("hams:", np.unique(ham_cluster[:, 0], return_counts=True))
print("spams:", np.unique(spam_cluster[:, 0], return_counts=True))
print("unknowns:", np.unique(unk_cluster[:, 0], return_counts=True))

print('\n\n\n\n')

from scipy.optimize import minimize_scalar

def objective_function(x):
    return 3 * x ** 4 - 2 * x + 1
res = minimize_scalar(objective_function)

import numpy as np
from scipy.optimize import minimize, LinearConstraint

n_buyers = 10
n_shares = 15

np.random.seed(10)

prices = np.random.random(n_buyers)
money_available = np.random.randint(1, 4, n_buyers)
n_shares_per_buyer = money_available / prices

print(prices, money_available, n_shares_per_buyer, sep="\n")
constraint = LinearConstraint(np.ones(n_buyers), lb=n_shares, ub=n_shares)
bounds = [(0, n) for n in n_shares_per_buyer]

def objective_function(x, prices):
    return -x.dot(prices)

res = minimize(
    objective_function,
    x0=10 * np.random.random(n_buyers),
    args=(prices,),
    constraints=constraint,
    bounds=bounds,
)
print("The total number of shares is:", sum(res.x))
print("Leftover money for each buyer:", money_available - res.x * prices)


