# Importing the script
from AllDistribution import Distribution, DistributionAnalyzer, OutputFormatter

#___________________________ Random Data ___________________________

# Uncomment and use one of the following distributions if desired

# data = list(np.random.gamma(2, 2, 5000))
# data = list(np.random.exponential(1.0, 5000))
# data = list(np.random.normal(0, 1, 5000))
# data = list(np.random.lognormal(0, 1, 5000))
# data = list(np.random.beta(2.0, 5.0, 5000))
# data = list(np.random.binomial(10, 0.5, 5000))
# data = list(np.random.negative_binomial(5, 0.5, 5000))
# data = list(np.random.poisson(5, 5000))
# data = list(np.random.geometric(0.5, 5000))
# data = list(np.random.weibull(2, 5000))

#___________________________ Self Data ___________________________

# Uncomment and provide file name to read data from file
# data = []
# with open("../Data/1.csv", 'r') as file:
#     for line in file:
#         value = eval(line.strip())
#         data.append(value)

#________________ Creating Object and Passing Data for distribution ________________

dis_obj = Distribution(data)

#________________ Checking which distribution is best ________________

# Uncomment to plot scatter

dis_obj.check_distribution()
# dis_obj.plot_data_scatter()

#___________________________ Call Particular Distribution ___________________________

# Uncomment and test a specific distribution

# itern = dis_obj.gamma_distribution(10)              # Testing pending
# itern = dis_obj.exponential_distribution(10)      # Testing pending
# itern = dis_obj.normal_distribution(10)           # Testing pending
# itern = dis_obj.log_normal_distribution(10)       # Testing pending
# itern = dis_obj.beta_distribution(10)             # Testing pending
# itern = dis_obj.binomial_distribution()           # Testing pending
# itern = dis_obj.negative_binomial_distribution()  # Testing pending
# itern = dis_obj.poisson_distribution()            # Testing pending
# itern = dis_obj.geometric_distribution()          # Testing pending
# itern = dis_obj.weibull_distribution(10)          # Testing pending

#___________________________ Getting Degree of Freedom ___________________________

dof = itern[1][list(itern[1])[-1]]

#_____________ Creating Object for OutputFormatter and Passing parameters _____________

out_obj = OutputFormatter(itern[0], data)

#_______________ Printing All Stats About Data with selected Distribution _______________


for k, v in itern[1].items():
    print(k, v)

# Uncomment to print all stats - mean variance std cv and all...   
out_obj.print_stats()

#_______________ Printing Chi Square Table of selected Distribution _______________

# Uncoment to print chi square table
# out_obj.print_chi_square_table()

#_______________ Checking selected Distribution is fitted or not _______________

# i pass 0.05 level of significance you can pass by your choice

chi_table = out_obj.find_chi_sqr_table_value(dof, 0.05)
out_obj.check_chi_sqr_value(chi_table)

#_______________ Visualizing selected Distribution _______________

# Uncoment to plot graphs

# out_obj.plot_continuous_bar()
# out_obj.plot_observed_vs_expected_frequency()
