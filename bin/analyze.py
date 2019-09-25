#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math

import time
import glob
import os

### CONSTANTS

RESULTS    = "results/*.csv"
IMAGE_DIR  = "images"

base_solver=""


### PLOTING HELPERS

def scatterplot(ax, x_data, y_data, label):
    # Plot the data, set the size (s), color and transparency (alpha) of the points
    ax.scatter(x_data, y_data, s = 10, alpha = 0.75, label = label)


def groupedbarplot(ax, x_data, y_data_list, y_data_names):
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual bar
    ind_width   = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration  = np.arange(-(total_width/2), total_width/2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.bar(x_data + alteration[i], y_data_list[i], label = y_data_names[i], width = ind_width)


### PLOTTING FUNCTIONS

def plot_cactus(data, name, show_date=False, yscale_log=True, out_type="pdf"):
    x_label = "Instance #"
    y_label = "Time (s)"
    title   = "Cactus: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")
    
    # Create the plot object
    fig, ax = plt.subplots()

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for solver, runs in data.items():
        flt = [r[2] for r in runs if r[1] in ['sat', 'unsat']]
        scatterplot(ax, list(range(len(flt))), sorted(flt), solver)

    ax.legend()
    fig.savefig("%s/%s"%(IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)


def plot_times(data, name, average=True, include_overall=False, show_date=False, out_type="pdf"):
    y_label = "Average Time (s)" if average else "Time (s)"
    title   = "Times: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")
    
    # Create the plot object
    fig, ax = plt.subplots()    

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices     = ["sat", "unsat", "unknown", "error", "overall"]
    choices     = choices if include_overall else choices[:-1]
    x_data      = list(range(len(choices)))
    y_data_list = []
    solvers     = []
    '''
    cost_eff_runs = data["i-uip-cost-eff"]
    one_uip_data= data["1-uip"]
    comparable_data_cost =[]
    compareable_one = []
    comparable_data_cost_size =[]
    compareable_one_size = []
    for i in range(len(cost_eff_runs)):
        if  not cost_eff_runs[i][1].startswith("timeout") and not one_uip_data[i][1].startswith("timeout"):
            if (cost_eff_runs[i][2] <= 120 and one_uip_data[i][2] <= 120):
                comparable_data_cost.append(cost_eff_runs[i][2])
                compareable_one.append(one_uip_data[i][2])
                comparable_data_cost_size.append(cost_eff_runs[i][3])
                compareable_one_size.append(one_uip_data[i][3])

    print (np.sum(comparable_data_cost) / len(comparable_data_cost))
    print(np.sum(compareable_one) / len(compareable_one))
    print (np.sum(comparable_data_cost_size) / len(comparable_data_cost_size))
    print(np.sum(compareable_one_size) / len(compareable_one_size))

    plt.plot(comparable_data_cost,compareable_one, 'ro')
    x = np.arange(0,540)
    plt.plot(x, x)
    plt.axis([0, 120, 0 ,120])
    plt.xlabel("i-uip-cost-eff")
    plt.ylabel("1-uip")
    plt.show()
    '''
    for solver, runs in data.items():
        solvers.append(solver)
        times = time_results(runs)

        if average:
            count = count_results(runs)
            count = [count[0], count[1], count[2], count[4]] #remove timeouts
            count = count + [sum(count)] if include_overall else count

            for i in range(len(count)):
                times[i] = times[i]/count[i] if count[i] > 0 else 0

        y_data_list.append(times if include_overall else times[:-1])

    groupedbarplot(ax, x_data, y_data_list, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s"%(IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_times(average, choices, solvers, y_data_list)


def plot_conflict_size(data, name, average=True, include_overall=False, show_date=False, out_type="pdf"):
    y_label = "Average conflict clause size " if average else "Clauses "
    title = "Clause size: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")

    # Create the plot object
    fig, ax = plt.subplots()

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    choices = choices if include_overall else choices[:-1]
    x_data = list(range(len(choices)))
    y_data_list = []
    solvers = []

    for solver, runs in data.items():
        solvers.append(solver)
        sizes ,count= conflict_size_Result(runs)

        if average:
            count = [count[0], count[1], count[2], count[3], count[4]]  # remove timeouts
            count = count + [sum(count)] if include_overall else count

            for i in range(len(count)):
                sizes[i] = sizes[i] / count[i] if count[i] > 0 else 0

        y_data_list.append(sizes if include_overall else sizes[:-1])

    groupedbarplot(ax, x_data, y_data_list, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s" % (IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_conflict_size(average, choices, solvers, y_data_list)

def plot_mem_use(data, name, average=True, include_overall=False, show_date=False, out_type="pdf"):
    y_label = " memory use " if average else "memory use "
    title = "memory use: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")

    # Create the plot object
    fig, ax = plt.subplots()

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    choices = choices if include_overall else choices[:-1]
    x_data = list(range(len(choices)))
    y_data_list = []
    solvers = []

    for solver, runs in data.items():
        solvers.append(solver)
        memory_use, count = memory_use_result(runs)

        if average:
            count = [count[0], count[1], count[2], count[3], count[4]]  # remove timeouts
            count = count + [sum(count)] if include_overall else count

            for i in range(len(count)):
                memory_use[i] = memory_use[i] / count[i] if count[i] > 0 else 0

        y_data_list.append(memory_use if include_overall else memory_use[:-1])

    groupedbarplot(ax, x_data, y_data_list, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s" % (IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_mem_use(average, choices, solvers, y_data_list)

def plot_reduction(data, name, average=True, include_overall=False, show_date=False, out_type="pdf"):
    y_label = " reduction percent " if average else "reduction percent "
    title = "reduction percent: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")

    # Create the plot object
    fig, ax = plt.subplots()

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    choices = choices if include_overall else choices[:-1]
    x_data = list(range(len(choices)))
    y_data_list = []
    solvers = []

    for solver, runs in data.items():
        solvers.append(solver)
        reduction, count = reduction_result(runs)

        if average:

            count = [count[0], count[1], count[2], count[3], count[4]]  # remove timeouts
            count = count + [sum(count)] if include_overall else count

            for i in range(len(count)):
                reduction[i] = reduction[i] / count[i] if count[i] > 0 else 0

        y_data_list.append(reduction if include_overall else reduction[:-1])

    groupedbarplot(ax, x_data, y_data_list, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s" % (IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_reduction(average, choices, solvers, y_data_list)


def plot_average_len(data, name, average=True, include_overall=False, show_date=False, out_type="pdf"):
    y_label = " avg clause len" if average else " average clause len "
    title = "average clause size: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")

    # Create the plot object
    fig, ax = plt.subplots()

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    choices = choices if include_overall else choices[:-1]
    x_data = list(range(len(choices)))
    y_data_list = []
    solvers = []
    '''
    one_uip = data["i-uip-mini_sat15"]
    all_uip = data["i-uip-mini-greedy_sat15"]
    comparable_data_cost = []
    compareable_one = []
    regular_behavior = 0
    for i in range(len(all_uip)):
        i_uip_data = all_uip[i][4]
        one_uip_data = one_uip[i][4]
        if (i_uip_data >= one_uip_data and all_uip[i][2]>= one_uip[i][2]):
            regular_behavior+=1
        elif (i_uip_data < one_uip_data and all_uip[i][2] < one_uip[i][2]):
            regular_behavior+=1


        comparable_data_cost.append(i_uip_data)
        compareable_one.append(one_uip_data)

    fraction =  np.round(np.divide(comparable_data_cost, compareable_one) * 100) - 100
    print (np.sum( np.greater_equal(compareable_one, comparable_data_cost)) / len(compareable_one))
    print ("regular behvaior percentage %f" %  float(regular_behavior / len(compareable_one)))
    plt.hist(fraction, normed=True, cumulative=False, label='PDF',
             histtype='step', alpha=0.8, color='k')
    #x = np.arange(0, 100, 0.01)
    #plt.plot(x, x)
    #plt.axis([0, 100, 0, 100])
    #plt.title("conflict per decision")
    #plt.xlabel("all-uip")
    #plt.ylabel("1-uip")
    plt.show()
    '''
    for solver, runs in data.items():
        solvers.append(solver)
        average_len, count = average_len_result(runs)

        if average:
            count = [count[0], count[1], count[2], count[3], count[4]]  # remove timeouts
            count = count + [sum(count)] if include_overall else count

            for i in range(len(count)):
                average_len[i] = average_len[i] / count[i] if count[i] > 0 else 0

        y_data_list.append(average_len if include_overall else average_len[:-1])

    groupedbarplot(ax, x_data, y_data_list, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s" % (IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_average_len(average, choices, solvers, y_data_list)

def plot_i_uip_reduction(data, name, average=True, include_overall=False, show_date=False, out_type="pdf"):
    y_label = " avg i_uip_reduction" if average else " i_uip_reduction "
    title = "average i_uip_reduction: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")
    '''
    all_uip = data["i-uip-lbd_easy-medium"]
    one_uip = data["1-uip_easy-medium"]
    comparable_data_cost = []
    compareable_one = []
    for i in range(len(all_uip)):
                comparable_data_cost.append(all_uip[i][5])
                compareable_one.append(one_uip[i][5])


    plt.plot(comparable_data_cost, compareable_one, 'ro')
    x = np.arange(0, 1.5, 0.01)
    plt.plot(x, x)
    plt.axis([0, 1, 0, 1])
    plt.title("conflict per decision")
    plt.xlabel("all-uip")
    plt.ylabel("1-uip")
    plt.show()
    '''
    # Create the plot object
    fig, ax = plt.subplots()

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    choices = choices if include_overall else choices[:-1]
    x_data = list(range(len(choices)))
    y_data_list = []
    solvers = []

    for solver, runs in data.items():
        solvers.append(solver)
        i_uip_reduction, count = i_uip_reduction_result(runs)

        if average:
            count = [count[0], count[1], count[2], count[3], count[4]]  # remove timeouts
            count = count + [sum(count)] if include_overall else count

            for i in range(len(count)):
                i_uip_reduction[i] = i_uip_reduction[i] / count[i] if count[i] > 0 else 0

        y_data_list.append(i_uip_reduction if include_overall else i_uip_reduction[:-1])

    groupedbarplot(ax, x_data, y_data_list, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s" % (IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_i_uip_reduction(average, choices, solvers, y_data_list)


def plot_counts(data, name, show_date=False, out_type="pdf"):
    y_label = "# Occurences"
    title   = "Counts: %s" % " vs. ".join(solver for solver in data.keys())

    if show_date:
        title += " (%s)" % time.strftime("%d/%m/%Y")
    
    # Create the plot object
    fig, ax = plt.subplots()    

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_ylabel(y_label)

    choices = ["sat", "unsat", "unknown", "timeout", "error"]
    x_data  = list(range(len(choices)))
    counts  = []
    solvers = []

    for solver, runs in data.items():
        solvers.append(solver)
        counts.append(count_results(runs))

    groupedbarplot(ax, x_data, counts, solvers)
    ax.set_xticklabels(choices)
    ax.set_xticks(list(range(len(choices))))
    ax.legend()
    fig.savefig("%s/%s"%(IMAGE_DIR, "%s.%s" % (name, out_type)), bbox_inches='tight')
    plt.close(fig)

    print_counts(choices, solvers, counts)


### ANALYSIS AND AGGREGATION

def count_results(runs):
    choices = ["sat", "unsat", "unknown", "timeout", "error"]
    results = [0 for x in choices]

    for r in runs["Result"]:
        if r == "sat":
            results[0] += 1
        if r == "unsat":
            results[1] += 1
        if r == "unknown":
            results[2] += 1
        if "timeout" in r:
            results[3] += 1
        if r == "error":
            results[4] += 1

    return results


def time_results(runs):
    choices = ["sat", "unsat", "unknown", "error", "overall"]
    results = [0 for x in choices]

    for r in range(len(runs["Result"])):
        if runs["Result"][r] == "sat":
            results[0] += runs["Time"][r]
        if runs["Result"][r] == "unsat":
            results[1] += runs["Time"][r]
        if runs["Result"][r] == "unknown":
            results[2] += runs["Time"][r]
        if runs["Result"][r] == "error":
            results[3] += runs["Time"][r]
        results[4] += runs["Time"][r]

    return results

def conflict_size_Result(runs):
    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    results = [0 for x in choices]
    count = [0,0,0,0,0,0]

    for r in range(len(runs["conflict_size"])):
        if (math.isnan(float(runs["conflict_size"][r]))):
            continue
        if runs["Result"][r] == "sat":
            results[0] += float(runs["conflict_size"][r])
            count[0] +=1
        if runs["Result"][r] == "unsat":
            results[1] += float(runs["conflict_size"][r])
            count[1] += 1
        if runs["Result"][r] == "unknown":
            results[2] += float(runs["conflict_size"][r])
            count[2] += 1
        if runs["Result"][r] == "error":
            results[4] += float(runs["conflict_size"][r])
            count[4] += 1
        if runs["Result"][r].startswith("timeout"):
            results[3] += float(runs["conflict_size"][r])
            count[3] += 1
        results[5] += float(runs["conflict_size"][r])
        count[5] += 1

    return results, count


def memory_use_result(runs):
    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    results = [0 for x in choices]
    count = [0,0,0,0,0,0]

    for r in range(len(runs["mem_use"])):
        if (math.isnan(float(runs["mem_use"][r]))):
            continue
        if runs["Result"][r] == "sat":
            results[0] += float(runs["mem_use"][r])
            count[0] +=1
        if runs["Result"][r] == "unsat":
            results[1] += float(runs["mem_use"][r])
            count[1] += 1
        if runs["Result"][r] == "unknown":
            results[2] += float(runs["mem_use"][r])
            count[2] += 1
        if runs["Result"][r] == "error":
            results[4] += float(runs["mem_use"][r])
            count[4] += 1
        if runs["Result"][r].startswith("timeout"):
            results[3] += float(runs["mem_use"][r])
            count[3] += 1
        results[5] += float(runs["mem_use"][r])
        count[5] += 1

    return results, count

def reduction_result(runs):
    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    results = [0 for x in choices]
    count = [0,0,0,0,0,0]

    for r in range(len(runs["reduction"])):
        if (math.isnan(float(runs["reduction"][r]))):
            continue
        if runs["Result"][r] == "sat":
            results[0] += float(runs["reduction"][r])
            count[0] +=1
        if runs["Result"][r] == "unsat":
            results[1] += float(runs["reduction"][r])
            count[1] += 1
        if runs["Result"][r] == "unknown":
            results[2] += float(runs["reduction"][r])
            count[2] += 1
        if runs["Result"][r] == "error":
            results[4] += float(runs["reduction"][r])
            count[4] += 1
        if runs["Result"][r].startswith("timeout"):
            results[3] += float(runs["reduction"][r])
            count[3] += 1
        results[5] += float(runs["reduction"][r])
        count[5] += 1

    return results, count

def i_uip_reduction_result(runs):
    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    results = [0 for x in choices]
    count = [0,0,0,0,0,0]

    for r in range(len(runs["i_uip_reduction"])):
        if (math.isnan(float(runs["i_uip_reduction"][r]))):
            continue
        if runs["Result"][r] == "sat":
            results[0] += float(runs["i_uip_reduction"][r])
            count[0] +=1
        if runs["Result"][r] == "unsat":
            results[1] += float(runs["i_uip_reduction"][r])
            count[1] += 1
        if runs["Result"][r] == "unknown":
            results[2] += float(runs["i_uip_reduction"][r])
            count[2] += 1
        if runs["Result"][r] == "error":
            results[4] += float(runs["i_uip_reduction"][r])
            count[4] += 1
        if runs["Result"][r].startswith("timeout"):
            results[3] += float(runs["i_uip_reduction"][r])
            count[3] += 1
        results[5] += float(runs["i_uip_reduction"][r])
        count[5] += 1

    return results, count

def average_len_result(runs):
    choices = ["sat", "unsat", "unknown", "error", "timeout", "overall"]
    results = [0 for x in choices]
    count = [0,0,0,0,0,0]

    for r in range(len(runs["average_len"])):
        if (math.isnan(float(runs["average_len"][r]))):
            continue
        if runs["Result"][r] == "sat":
            results[0] += float(runs["average_len"][r])
            count[0] +=1
        if runs["Result"][r] == "unsat":
            results[1] += float(runs["average_len"][r])
            count[1] += 1
        if runs["Result"][r] == "unknown":
            results[2] += float(runs["average_len"][r])
            count[2] += 1
        if runs["Result"][r] == "error":
            results[4] += float(runs["average_len"][r])
            count[4] += 1
        if runs["Result"][r].startswith("timeout"):
            results[3] += float(runs["average_len"][r])
            count[3] += 1
        results[5] += float(runs["average_len"][r])
        count[5] += 1

    return results, count


def check_consensus(data):
    # ASSUMING IN SAME ORDER!!!
    issues     = []
    min_solved = min(len(runs) for solver, runs in data.items())

    for i in range(min_solved):
        votes = {}

        for solver, runs in data.items():
            votes[solver] = runs["Result"][i]
            problem       = runs['Instance'][i]

        done = False
        for _, va in votes.items():
            if done:
                break
            for _, vb in votes.items():
                if done:
                    break
                if va != vb and va in ['sat', 'unsat'] and vb in ['sat', 'unsat']:
                    issues.append((problem, votes))
                    done = True
                    break

    print_consensus_issues(issues)


### PRINTING RESULTS

def print_consensus_issues(issues):
    if len(issues) == 0:
        return

    print("\nDisagreements (%d):" % len(issues))
    print("Instance,", ", ".join(solver for solver in issues[0][1].keys()))

    for i in issues:
        print("%s," % i[0], ", ".join(i[1][solver] for solver in i[1].keys()))


def print_counts(choices, solvers, counts):
    print("\nCounts:")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(counts)):
        print(", ".join(c for c in [solvers[i]] + list(map(str, counts[i]))))


def print_times(average, choices, solvers, times):
    print("\nAverage Times (s):") if average else print("\nTimes (s):")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(times)):
        print(", ".join(c for c in [solvers[i]] + list(map(repr, times[i]))))

def print_conflict_size(average, choices, solvers, sizes):
    print("\nAverage conflict clause size :") if average else print("\nClause size :")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(sizes)):
        print(", ".join(c for c in [solvers[i]] + list(map(repr, sizes[i]))))

def print_mem_use(average, choices, solvers, sizes):
    print("\nAverage memory use :") if average else print("\nMemory use :")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(sizes)):
        print(", ".join(c for c in [solvers[i]] + list(map(repr, sizes[i]))))


def print_reduction(average, choices, solvers, sizes):
    print("\nAverage reduction  :") if average else print("\nreduction :")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(sizes)):
        print(", ".join(c for c in [solvers[i]] + list(map(repr, sizes[i]))))



def print_i_uip_reduction(average, choices, solvers, sizes):
    print("\nAverage i_uip_reduction  :") if average else print("\ni_uip_reduction :")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(sizes)):
        print(", ".join(c for c in [solvers[i]] + list(map(repr, sizes[i]))))

def print_average_len(average, choices, solvers, sizes):
    print("\nAverage average_len  :") if average else print("\naverage_len :")
    print("solver,", ", ".join(c for c in choices))

    for i in range(len(sizes)):
        print(", ".join(c for c in [solvers[i]] + list(map(repr, sizes[i]))))

#### ENTRY POINT
def main():
    global base_solver
    data         = {}
    result_files = glob.glob(RESULTS)

    for result in result_files:
        solver       = os.path.basename(result)[:-len(".csv")]
        if (solver.startswith("1-uip")):
            base_solver = solver
        data[solver] = np.genfromtxt(result, comments="OMG_WTF", delimiter=',', dtype=None, encoding=None, names=["Instance", "Result", "Time", "conflict_size", "average_len", "reduction", "i_uip_reduction", "mem_use", "core_clause", "lbd", "i_uip_attempt","i_uip_sucess"], skip_header=1)
        data[solver] = np.array(sorted(data[solver], key=lambda entry: entry[0]))
    
    check_consensus(data)
    plot_cactus(data, "overall_cactus", out_type="png")
    plot_counts(data, "overall_counts", out_type="png")
    plot_times(data, "overall_times", out_type="png")
    plot_conflict_size(data, "overall_conflict_size", out_type="png")
    plot_mem_use(data, "overall_mem_use", out_type="png")
    plot_reduction(data, "overall_reduction", out_type="png")
    plot_i_uip_reduction(data, "overall_i_uip_reduction", out_type="png")
    plot_average_len(data, "overall_average_len", out_type="png")


if __name__ == '__main__':
    main()
