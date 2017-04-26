#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    all_data = []
    r_sq = []
    ### your code goes here

    # find the sqared error i.e. sq(actual - prediction) for all
    for p, a, n in zip(predictions, ages, net_worths):
        r_sq.append((n - p)**2)
        all_data.append((p, a, n, (n - p)**2))

    # eliminate the highest 9 => 9 should be obtained by 10% of data formula
    r_sq.sort()
    to_be_excluded = int(len(all_data)*0.1)
    remove_list = r_sq[-to_be_excluded:]
    #(array([ 314.65206822]), array([57]), array([ 338.08951849]), array([ 549.31407506]))
    for x in all_data:
        if x[3] not in remove_list:
            cleaned_data.append((x[1], x[2], x[3]))

    # return the list with tuples that are rest 90%
    return cleaned_data

