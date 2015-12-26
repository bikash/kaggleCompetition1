

import sqlite3
import pandas as pd
from haversine import haversine

north_pole = (90,0)
weight_limit = 1000.0

def bb_sort(ll):
    s_limit = 5000
    optimal = False
    ll = [[0,north_pole,10]] + ll[:] + [[0,north_pole,10]]
    while not optimal:
        optimal = True
        for i in range(1,len(ll) - 2):
            lcopy = ll[:]
            lcopy[i], lcopy[i+1] = ll[i+1][:], ll[i][:]
            if path_opt_test(ll[1:-1])[0] > path_opt_test(lcopy[1:-1])[0]:
                print("Sort Swap")
                ll = lcopy[:]
                optimal = False
                s_limit -= 1
                if s_limit < 0:
                    optimal = True
                    break
    return ll[1:-1]

def prev_path_opt(curr,prev):
    curr = [[0,north_pole,10]] + curr[:] + [[0,north_pole,10]]
    prev = [[0,north_pole,10]] + prev[:] + [[0,north_pole,10]]
    for curr_ in range(1,len(curr) - 1):
        for prev_ in range(1,len(prev) - 1):
            #Test 1 - Swap
            lcopy_curr = curr[:]
            lcopy_prev = prev[:]
            lcopy_curr[curr_], lcopy_prev[prev_] = lcopy_prev[prev_][:], lcopy_curr[curr_][:]
            if ((path_opt_test(lcopy_curr[1:-1])[0] + path_opt_test(lcopy_prev[1:-1])[0]) < (path_opt_test(curr[1:-1])[0] + path_opt_test(prev[1:-1])[0])) and path_opt_test(lcopy_curr[1:-1])[2] <=1000 and path_opt_test(lcopy_prev[1:-1])[2] <= 1000:
                print("Trip Swap")
                curr = lcopy_curr[:]
                prev = lcopy_prev[:]
    return [curr[1:-1], prev[1:-1]]

def prev_path_opt_s1(curr,prev):
    curr = [[0,north_pole,10]] + curr[:] + [[0,north_pole,10]]
    prev = [[0,north_pole,10]] + prev[:] + [[0,north_pole,10]]
    for curr_ in range(1,len(curr) - 1):
        for prev_ in range(1,len(prev) - 1):
            #Test 2 - Swap to current
            lcopy_curr = curr[:]
            lcopy_prev = prev[:]
            #print("current: ",curr_,len(curr),len(lcopy_curr),"previous: ",prev_,len(prev),len(lcopy_prev))
            if len(lcopy_prev)-1 <= prev_:
                break
            lcopy_curr = lcopy_curr[:curr_+1][:] + [lcopy_prev[prev_]] + lcopy_curr[curr_+1:][:]
            lcopy_prev.pop(prev_)
            if ((path_opt_test(lcopy_curr[1:-1])[0] + path_opt_test(lcopy_prev[1:-1])[0]) <= (path_opt_test(curr[1:-1])[0] + path_opt_test(prev[1:-1])[0])) and path_opt_test(lcopy_curr[1:-1])[2] <=1000 and path_opt_test(lcopy_prev[1:-1])[2] <= 1000:
                print("Trip Swap - Give to current")
                curr = lcopy_curr[:]
                prev = lcopy_prev[:]
    return [curr[1:-1], prev[1:-1]]

def path_opt_test(llo):
    f_ = 0.0
    d_ = 0.0
    we_ = 0.0
    l_ = north_pole
    for i in range(len(llo)):
        d_ += haversine(l_, llo[i][1])
        we_ += llo[i][2]
        f_ += d_ * llo[i][2]
        l_ = llo[i][1]
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10 #sleigh weight for whole trip
    return [f_,d_,we_]

for n in range(2):
    gifts = pd.read_csv("data/gifts.csv").fillna(" ")
    subvers = pd.read_csv("output/submission_v" + str(n) +".csv").fillna(" ")
    c = sqlite3.connect(":memory:")
    gifts.to_sql("gifts",c)
    subvers.to_sql("vers",c)

    ou_ = open("output/submission_v" + str(n+1) + ".csv","w")
    ou_.write("TripId,GiftId\n")
    bm = 0.0
    b_ = 0.0
    first_trip = []
    previous_trip = []
    submission = pd.read_sql("SELECT TripId FROM vers GROUP BY TripId ORDER BY TripId;", c)
    for s_ in range(len(submission.TripId)):
        trip = pd.read_sql("SELECT gifts.GiftId, Latitude, Longitude, Weight FROM gifts INNER JOIN vers ON gifts.GiftId = vers.GiftId WHERE vers.TripId = " + str(submission.TripId[s_]) + " ORDER BY vers.[index] ASC;",c)
        b = []
        for x_ in range(len(trip.GiftId)):
            b.append([trip.GiftId[x_],(trip.Latitude[x_],trip.Longitude[x_]),trip.Weight[x_]])

       #OPTIMIZE PATH
        b = bb_sort(b)
        if s_ > 0:
            previous_trip, b = prev_path_opt_s1(previous_trip, b)
            previous_trip, b = prev_path_opt(previous_trip, b)
            previous_trip = bb_sort(previous_trip)
            #output
            print(submission.TripId[s_-1], b_, bm)
            if s_ > 1:
                b_ = path_opt_test(previous_trip)[0]
                bm += b_
                for x_ in range(len(previous_trip)):
                    ou_.write(str(submission.TripId[s_-1])+","+str(previous_trip[x_][0])+"\n")
        if s_ == 1:
            first_trip = previous_trip[:]
        previous_trip = b[:]

    #Now count the last trip after possible optimization and print the last two
    b = first_trip[:]
    previous_trip, b = prev_path_opt_s1(previous_trip, b)
    previous_trip, b = prev_path_opt(previous_trip, b)
    previous_trip = bb_sort(previous_trip)
    #Tally last one
    b_ = path_opt_test(previous_trip)[0]
    #print(submission.TripId[s_], b_)
    bm += b_
    #Tally first one
    b = bb_sort(b)
    b_ = path_opt_test(b)[0]
    #print(1, b_)
    bm += b_

    #output previous
    for x_ in range(len(previous_trip)):
        ou_.write(str(submission.TripId[s_])+","+str(previous_trip[x_][0])+"\n")
    #output first
    for x_ in range(len(b)):
        ou_.write(str(submission.TripId[0])+","+str(b[x_][0])+"\n")
    ou_.close()
    c.close()

    benchmark = 12469095701
    if bm < benchmark:
        print(n, "Improvement", bm, bm - benchmark, benchmark)
    else:
        print(n, "Try again", bm, bm - benchmark, benchmark)
    benchmark = float(bm)
