# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

def get_times(file_path):
    with open(file_path, 'r') as f:
        txt_lines = f.readlines()
        times = []
        for line in txt_lines:
            line_list = line.split(', ')
            for i in range(4):
                times.append(float(line_list[i*2]))
        return times


times = get_times('test_amp_vx.txt')

len(times)

for i in range(len(times)-1):
    print((times[i+1] - times[i]) - times[1])


