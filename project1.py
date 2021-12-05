import numpy as np
import time
import matplotlib.pyplot as plt

# n is array which stores number 5000,10000, ....., 30000
n = np.array([i for i in range(5000,35000,5000)])   

# generating random numbers upto n where each number is generated 'times'
# times and average is taken.
def generate_random_numbers(n,high = None):
    if high is None:
        high = n
    # First generate array of size n
    r = np.random.randint(low = 1,high = high,size = n)

    # return the array
    return r

# Implementing INSERTION SORT
def insertion_sort(A):
    # copy initial array so that the initial array is not changed
    sorting = A.copy()
    start = time.time()
    for j in range(1,len(sorting)):
        # store next position to compare on key
        key = sorting[j]
        # store the current position index on i
        i = j - 1
        # if the current index > 0 and, current value is greater than next value (key)
        while i >= 0 and sorting[i] > key:
            # next value = current value
            sorting[i + 1] = sorting[i]
            # decrease the indices one more.
            i = i - 1

        # if the while loop is finished, store the key on 'i + 1' place
        sorting[i + 1] = key
    end = time.time()
    
    return sorting,((end-start))

# Implementing SELECTION SORT
def selection_sort(A):
    # copy initial array so that the initial array is not changed
    sorting = A.copy()
    # 
    start = time.time()
    for i in range(len(sorting)):
        min_index = i
        for j in range(i + 1, len(A)):
            if sorting[j] < sorting[min_index] and j != min_index:
                min_index = j
        sorting[i],sorting[min_index] = sorting[min_index],sorting[i]
    end = time.time()
    return sorting,(end-start)

# implement swapping of random elements in an array 50 times. 
# used for input 4 
def exchange_50_times(A):
    # copy the original dictionary so nothing gets changed
    check = A.copy()
    n_count = 0
    for key in check.keys():
        for i in range(50):
            # generate two random integers according to number of element in the array
            r_1 = np.random.randint(1,n[n_count])
            r_2 = np.random.randint(1,n[n_count])
            # exchange the numbers in the indices
            check[key][r_1],check[key][r_2] = check[key][r_2],check[key][r_1]
        n_count += 1
    return check

# plot the figure
def plot_figure(time_ss,time_is,title):
    # select x_axis as the number of sample
    x_axis = n
    # find average time for 3 sample of insertion sort
    y_axis_time_is = time_is
    y_axis_time_ss = time_ss 

    fig = plt.figure(figsize=(8, 6), dpi=80)
    fig.suptitle(title)
    plt.xlabel("Input array size")
    plt.ylabel("time taken (sec)")
    # plot for insertion sort
    plt.plot(x_axis,y_axis_time_is,'bo',label = "insertion sort")
    plt.plot(x_axis,y_axis_time_is,'b')
    # plot for selection sort
    plt.plot(x_axis,y_axis_time_ss,'ko',label = "selection sort")
    plt.plot(x_axis,y_axis_time_ss,'k')
    plt.legend()
    fig.savefig(title+'.jpg')
    


###################################################################################
# INPUT1 / PLOT 1
#
# Generate each xi to be a uniformly random integer between 1 and n. On random
# inputs that you generate: For each data point take average of 3 runs and plot
# the result.
###################################################################################
def input1_plot1():

    # Create 3 sample of input 1 ############################################
    sample1_input1 = {}
    sample2_input1 = {}
    sample3_input1 = {}

    # Generate random inputs
    for i in range(len(n)):
        sample1_input1["sample1_"+str(n[i])] = generate_random_numbers(n[i])
    for i in range(len(n)):
        sample2_input1["sample2_"+str(n[i])] = generate_random_numbers(n[i])
    for i in range(len(n)):
        sample3_input1["sample3_"+str(n[i])] = generate_random_numbers(n[i])
    ######################### Input 1 created################################


    # dictionary for storing time taken for selection sort(ss) to sort sample1,sample2, and sample3
    sample1_input1_time_ss = {}
    sample2_input1_time_ss = {}
    sample3_input1_time_ss = {}

    # dictionary for storing time taken for insertion sort(is) to sort sample1,sample2,and sample3
    sample1_input1_time_is = {}
    sample2_input1_time_is = {}
    sample3_input1_time_is = {}

    # dictionary for storing sorted output of sample1, sample2, sample3 using selection sort
    sample1_output1_ss = {}
    sample2_output1_ss = {}
    sample3_output1_ss = {}

    #dictionary for storing sorted output of sample1,sample2,sample3 using insertion sort
    sample1_output1_is = {}
    sample2_output1_is = {}
    sample3_output1_is = {}

    # run insertion & selection sort for all the values in n for sample 1
    for i in range(len(n)):
        # run selection sort for all sample1
        sample1_output1_ss["sample1_"+str(n[i])],sample1_input1_time_ss["sample1_"+str(n[i])] = selection_sort(sample1_input1["sample1_"+str(n[i])])
        # run insertion sort for all sample1
        sample1_output1_is["sample1_"+str(n[i])],sample1_input1_time_is["sample1_"+str(n[i])] = insertion_sort(sample1_input1["sample1_"+str(n[i])])

    # run insertion & selection sort for all the values in n for sample 2
    for i in range(len(n)):
        # run selection sort for all sample1
        sample2_output1_ss["sample2_"+str(n[i])],sample2_input1_time_ss["sample2_"+str(n[i])] = selection_sort(sample2_input1["sample2_"+str(n[i])])
        # run insertion sort for all sample1
        sample2_output1_is["sample2_"+str(n[i])],sample2_input1_time_is["sample2_"+str(n[i])] = insertion_sort(sample2_input1["sample2_"+str(n[i])])
        
    # run insertion & selection sort for all the values in n for sample 3
    for i in range(len(n)):
        # run selection sort for all sample1
        sample3_output1_ss["sample3_"+str(n[i])],sample3_input1_time_ss["sample3_"+str(n[i])] = selection_sort(sample3_input1["sample3_"+str(n[i])])
        # run insertion sort for all sample1
        sample3_output1_is["sample3_"+str(n[i])],sample3_input1_time_is["sample3_"+str(n[i])] = insertion_sort(sample3_input1["sample3_"+str(n[i])])

    # print(sample2_input1_time_is)
    # print(sample2_input1_time_ss)
    # average time by 3 samples stored in this array
    time_ss = np.zeros(6)
    time_is = np.zeros(6)

    for i in range(len(n)):
        time_ss[i] = sample1_input1_time_ss["sample1_"+str(n[i])] + sample2_input1_time_ss["sample2_"+str(n[i])] + sample3_input1_time_ss["sample3_"+str(n[i])]
        time_ss[i] = time_ss[i] / 3

        time_is[i] = sample1_input1_time_is["sample1_"+str(n[i])] + sample2_input1_time_is["sample2_"+str(n[i])] + sample3_input1_time_is["sample3_"+str(n[i])]
        time_is[i] = time_is[i] / 3


    plot_figure(time_ss, time_is, title = "plot1")

###################################################################################
# INPUT2 / PLOT 2
#
# Generate each xi to be a uniformly random integer between 1 and n and sort the
# resulting sequence in non-decreasing order. Then run each of the sorting algorithms
# again and measure its performance
###################################################################################
def input2_plot2():
    ############################## Create 3 sample of input 2 #####################
    sample1_input2 = {}
    sample2_input2 = {}
    sample3_input2 = {}

    # Generate random inputs
    for i in range(len(n)):
        sample1_input2["sample1_"+str(n[i])] = generate_random_numbers(n[i])
    # for i in range(len(n)):
    #     sample2_input2["sample2_"+str(n[i])] = generate_random_numbers(n[i])
    # for i in range(len(n)):
    #     sample3_input2["sample3_"+str(n[i])] = generate_random_numbers(n[i])

    # Sort the input
    for key in sample1_input2.keys():
        sample1_input2[key] = np.sort(sample1_input2[key])
    
    # for key in sample2_input2.keys():
    #     sample2_input2[key] = np.sort(sample2_input2[key])
    
    # for key in sample3_input2.keys():
    #     sample3_input2[key] = np.sort(sample3_input2[key])
    ###############################################################################

    # dictionary for storing time taken for selection sort(ss) to sort sample1,sample2, and sample3
    sample1_input2_time_ss = {}
    # sample2_input2_time_ss = {}
    # sample3_input2_time_ss = {}

    # dictionary for storing time taken for insertion sort(is) to sort sample1,sample2,and sample3
    sample1_input2_time_is = {}
    # sample2_input2_time_is = {}
    # sample3_input2_time_is = {}

    # dictionary for storing sorted output of sample1, sample2, sample3 using selection sort
    sample1_output2_ss = {}
    # sample2_output2_ss = {}
    # sample3_output2_ss = {}

    #dictionary for storing sorted output of sample1,sample2,sample3 using insertion sort
    sample1_output2_is = {}
    # sample2_output2_is = {}
    # sample3_output2_is = {}

    # run insertion & selection sort for all the values in n for sample 1
    for key in sample1_input2.keys():
        # run selection sort for all sample1
        sample1_output2_ss[key],sample1_input2_time_ss[key] = selection_sort(sample1_input2[key])
        # run insertion sort for all sample1
        sample1_output2_is[key],sample1_input2_time_is[key] = insertion_sort(sample1_input2[key])

    # # run insertion & selection sort for all the values in n for sample 2
    # for key in sample2_input2.keys():
    #     # run selection sort for all sample1
    #     sample2_output2_ss[key],sample2_input2_time_ss[key] = selection_sort(sample2_input2[key])
    #     # run insertion sort for all sample1
    #     sample2_output2_is[key],sample2_input2_time_is[key] = insertion_sort(sample2_input2[key])
        
    # # run insertion & selection sort for all the values in n for sample 3
    # for key in sample3_input2.keys():
    #     # run selection sort for all sample1
    #     sample3_output2_ss[key],sample3_input2_time_ss[key] = selection_sort(sample3_input2[key])
    #     # run insertion sort for all sample1
    #     sample3_output2_is[key],sample3_input2_time_is[key] = insertion_sort(sample3_input2[key])

    # store the time taken by insertion sort and selection sort in array
    time_ss = np.zeros(6)
    time_is = np.zeros(6)

    for i in range(len(n)):
        time_ss[i] = sample1_input2_time_ss["sample1_"+str(n[i])]
        time_is[i] = sample1_input2_time_is["sample1_"+str(n[i])] 

    plot_figure(time_ss,time_is,title = "plot2")

###################################################################################
# INPUT3 / PLOT 3
#
# Generate each xi to be a uniformly random integer between 1 and n and sort the 
# resulting sequence in non-increasing order. Then run each of the sorting algorithms
# again and measure it's performance
###################################################################################
def input3_plot3():
    ##################### Create 3 sample of input 3 ##############################
    sample1_input3 = {}
    # sample2_input3 = {}
    # sample3_input3 = {}
    
    # Generate random inputs
    for i in range(len(n)):
        sample1_input3["sample1_"+str(n[i])] = generate_random_numbers(n[i])
    # for i in range(len(n)):
    #     sample2_input3["sample2_"+str(n[i])] = generate_random_numbers(n[i])
    # for i in range(len(n)):
    #     sample3_input3["sample3_"+str(n[i])] = generate_random_numbers(n[i])

    # Sort the inputs and reverse it
    for key in sample1_input3.keys():
        sample1_input3[key] = np.sort(sample1_input3[key])[::-1]
    
    # for key in sample2_input3.keys():
    #     sample2_input3[key] = np.sort(sample2_input3[key])[::-1]
    
    # for key in sample3_input3.keys():
    #     sample3_input3[key] = np.sort(sample3_input3[key])[::-1]

    ##############################################################################

    # dictionary for storing time taken for selection sort(ss) to sort sample1,sample2, and sample3
    sample1_input3_time_ss = {}
    # sample2_input3_time_ss = {}
    # sample3_input3_time_ss = {}

    # dictionary for storing time taken for insertion sort(is) to sort sample1,sample2,and sample3
    sample1_input3_time_is = {}
    # sample2_input3_time_is = {}
    # sample3_input3_time_is = {}

    # dictionary for storing sorted output of sample1, sample2, sample3 using selection sort
    sample1_output3_ss = {}
    # sample2_output3_ss = {}
    # sample3_output3_ss = {}

    #dictionary for storing sorted output of sample1,sample2,sample3 using insertion sort
    sample1_output3_is = {}
    # sample2_output3_is = {}
    # sample3_output3_is = {}

    # run insertion & selection sort for all the values in n for sample 1
    for key in sample1_input3.keys():
        # run selection sort for all sample1
        sample1_output3_ss[key],sample1_input3_time_ss[key] = selection_sort(sample1_input3[key])
        # run insertion sort for all sample1
        sample1_output3_is[key],sample1_input3_time_is[key] = insertion_sort(sample1_input3[key])

    # # run insertion & selection sort for all the values in n for sample 2
    # for key in sample2_input3.keys():
    #     # run selection sort for all sample1
    #     sample2_output3_ss[key],sample2_input3_time_ss[key] = selection_sort(sample2_input3[key])
    #     # run insertion sort for all sample1
    #     sample2_output3_is[key],sample2_input3_time_is[key] = insertion_sort(sample2_input3[key])
        
    # # run insertion & selection sort for all the values in n for sample 3
    # for key in sample3_input3.keys():
    #     # run selection sort for all sample1
    #     sample3_output3_ss[key],sample3_input3_time_ss[key] = selection_sort(sample3_input3[key])
    #     # run insertion sort for all sample1
    #     sample3_output3_is[key],sample3_input3_time_is[key] = insertion_sort(sample3_input3[key])

    time_ss = np.zeros(6)
    time_is = np.zeros(6)

    for i in range(len(n)):
        time_ss[i] = sample1_input3_time_ss["sample1_"+str(n[i])]
        time_is[i] = sample1_input3_time_is["sample1_"+str(n[i])] 

    plot_figure(time_ss,time_is,title = "plot3")


###################################################################################
# INPUT4 / PLOT 4
#
# Generate input as plot 2
# Repeat the following 50 times: pick two random integers i and j, and exchange xi 
# and xj
###################################################################################
def input4_plot4():
    ############### Create 3 sample of input 3 ####################################
    sample1_input4 = {}
    sample2_input4 = {}
    sample3_input4 = {}
    
    # Generate random inputs
    for i in range(len(n)):
        sample1_input4["sample1_"+str(n[i])] = generate_random_numbers(n[i])

    for i in range(len(n)):
        sample2_input4["sample2_"+str(n[i])] = generate_random_numbers(n[i])
        
    for i in range(len(n)):
        sample3_input4["sample3_"+str(n[i])] = generate_random_numbers(n[i])

    # Sort the inputs
    for key in sample1_input4.keys():
        sample1_input4[key] = np.sort(sample1_input4[key])
    
    for key in sample2_input4.keys():
        sample2_input4[key] = np.sort(sample2_input4[key])
    
    for key in sample3_input4.keys():
        sample3_input4[key] = np.sort(sample3_input4[key])

    # pick two random integers i and j, and exchange xi and xj
    sample1_input4 = exchange_50_times(sample1_input4)
    sample2_input4 = exchange_50_times(sample2_input4)
    sample3_input4 = exchange_50_times(sample3_input4)    
    ##############################################################################

    # dictionary for storing time taken for selection sort(ss) to sort sample1,sample2, and sample3
    sample1_input4_time_ss = {}
    sample2_input4_time_ss = {}
    sample3_input4_time_ss = {}

    # dictionary for storing time taken for insertion sort(is) to sort sample1,sample2,and sample3
    sample1_input4_time_is = {}
    sample2_input4_time_is = {}
    sample3_input4_time_is = {}

    # dictionary for storing sorted output of sample1, sample2, sample3 using selection sort
    sample1_output4_ss = {}
    sample2_output4_ss = {}
    sample3_output4_ss = {}

    #dictionary for storing sorted output of sample1,sample2,sample3 using insertion sort
    sample1_output4_is = {}
    sample2_output4_is = {}
    sample3_output4_is = {}

    # run insertion & selection sort for all the values in n for sample 1
    for key in sample1_input4.keys():
        # run selection sort for all sample1
        sample1_output4_ss[key],sample1_input4_time_ss[key] = selection_sort(sample1_input4[key])
        # run insertion sort for all sample1
        sample1_output4_is[key],sample1_input4_time_is[key] = insertion_sort(sample1_input4[key])

    # run insertion & selection sort for all the values in n for sample 2
    for key in sample2_input4.keys():
        # run selection sort for all sample1
        sample2_output4_ss[key],sample2_input4_time_ss[key] = selection_sort(sample2_input4[key])
        # run insertion sort for all sample1
        sample2_output4_is[key],sample2_input4_time_is[key] = insertion_sort(sample2_input4[key])
        
    # run insertion & selection sort for all the values in n for sample 3
    for key in sample3_input4.keys():
        # run selection sort for all sample1
        sample3_output4_ss[key],sample3_input4_time_ss[key] = selection_sort(sample3_input4[key])
        # run insertion sort for all sample1
        sample3_output4_is[key],sample3_input4_time_is[key] = insertion_sort(sample3_input4[key])

    time_ss = np.zeros(6)
    time_is = np.zeros(6)

    for i in range(len(n)):
        time_ss[i] = sample1_input4_time_ss["sample1_"+str(n[i])] + sample2_input4_time_ss["sample2_"+str(n[i])] + sample3_input4_time_ss["sample3_"+str(n[i])]
        time_ss[i] = time_ss[i] / 3

        time_is[i] = sample1_input4_time_is["sample1_"+str(n[i])] + sample2_input4_time_is["sample2_"+str(n[i])] + sample3_input4_time_is["sample3_"+str(n[i])]
        time_is[i] = time_is[i] / 3


    plot_figure(time_ss, time_is, title = "plot4")

###################################################################################
# INPUT5 
#
#
#
#
###################################################################################
def input5():
    # Generate input 5: 100,000 inputs ranging from 1 to 50
    input5 = generate_random_numbers(100000,high = 50)
    
    # sort the input5 using insertion sort
    output5_is, input5_time_is = insertion_sort(input5)

    # sort the input5 using selection sort
    output5_ss, input5_time_ss = selection_sort(input5)

    print("The time taken to sort input5 using insertion sort is :", input5_time_is," seconds")
    print("The time taken to sort input5 using selection sort is :", input5_time_ss," seconds")


if __name__ == "__main__":
    print("Plotting plot1")
    input1_plot1()
    print("Plotting plot2")
    input2_plot2()
    print("Plotting plot3")
    input3_plot3()
    print("Plotting plot4")
    input4_plot4()
    input5()

    # show plots altogether at once
    plt.show()