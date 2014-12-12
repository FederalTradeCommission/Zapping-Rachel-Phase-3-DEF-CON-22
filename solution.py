'''
Solution to Phase 3 for Zap Rachel
Author - Sean Beck
Email - seanmckaybeck@yahoo.com

After looking through the example data the best criteria
found were that during the 4-noon time period there were
very few robo calls. Next to none actually. So assuming
calls during that time are not robocalls is relatively
accurate. Also if the number called more than about 5 people
it is most likely a robo caller. There are of course exceptions
to this but it is mostly true. 

It became more difficult to differentiate robo and human when
a number only called one person one time. This did not happen
very often for robo callers but it did happen. 
'''
from pandas import read_csv


def save_entry(row, all_calls):
    if row['FROM'] in all_calls:
        if row['TO'] in all_calls[row['FROM']]:
            all_calls[row['FROM']][row['TO']] += 1
        else:
            all_calls[row['FROM']][row['TO']] = 1
    else:
        all_calls[row['FROM']] = {}
        all_calls[row['FROM']][row['TO']] = 1

    return all_calls


df = read_csv('FTC-DEFCON Data Set 2.csv')
possible_robo_numbers = []
not_robos = []
all_calls = {}

for index, row in df.iterrows():
    all_calls = save_entry(row, all_calls)
    # if it's between 4 and noon it probably is not a robocall
    call_date, call_time = row['DATE/TIME'].split(' ')
    hour, minutes = call_time.split(':')
    hour = int(hour)
    if hour > 3 and hour < 12:
        not_robos.append(row['FROM'])

# now check based on criteria
for number in all_calls:
    if len(all_calls[number]) > 5:
        # it *_probably_* is a robocall
        possible_robo_numbers.append(number)

not_robos = list(set(not_robos))
possible_robo_numbers = list(set(possible_robo_numbers))

robo_numbers = set()

for number in possible_robo_numbers:
    if number not in not_robos:
        robo_numbers.add(number)

for index, row in df.iterrows():
    if row['FROM'] in robo_numbers:
        df.loc[index, 'LIKELY ROBOCALL'] = 'X'

df.to_csv(path_or_buf='answers.csv')
