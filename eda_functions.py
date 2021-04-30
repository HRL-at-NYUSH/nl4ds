# ------------------------------------Import Libraries----------------------------------------

import os
import glob
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import statsmodels.api as sm
# import patsy
# import sklearn

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

global default_dpi
default_dpi = 90


# -------------------------------------Data Selection----------------------------------------------

def load_census(which_year, version_code = 'data_2021_may'):
  '''Provide the census year you will to load (choose from any year between 1850 - 1940, except for 1890)'''
  if str(which_year) == '1890':
    print('Unfortunately, census records for NYC in 1890 are not available because a fire destroyed the collection. Try other years between 1850 and 1940.')
    return None

  try:
    df = pd.read_csv('/content/drive/My Drive/'+version_code+'/census_' + str(which_year) + '.csv')
    from IPython.display import clear_output
    clear_output()
    print('\nThere are '+str(len(df))+' entries.\n')
    print('Available columns:\n\n')
    print_list(df.columns.tolist())
    print('\n\n')
    return df

  except FileNotFoundError as e:
    print('File Not Found! Please contact Tim at gw923@nyu.edu for acccess to certain datasets.')
  except NameError as e:
    print('Function not defined yet! Please check if you have run the first cell in this notebook.')


def check_parenthesis_and_replace_comma_within_parenthesis(string):

  output = []
  letters = list(string.strip())
  waiting_to_close = False
  while len(letters) > 0:
    head = letters.pop(0)
    if head == '[':
      if waiting_to_close == False:
        waiting_to_close = True
      else:
        return False
    elif head == ']':
      if waiting_to_close == True:
        waiting_to_close = False
      else:
        return False

    if (head == ',' or head == ':') and waiting_to_close == True:
      output.append('|')
    else:
      output.append(head)

  if waiting_to_close == False:
    return ''.join(output)
  else:
    return False

def replace_first_occurence_of_sign(string, sign, replacement):

  first_position_of_sign = string.index(sign)
  new_string = string[:first_position_of_sign] + replacement + string[first_position_of_sign + len(sign):]
  return new_string

def check_for_criteria_type(string, data, sign, alternative_sign, valid_cols):

  sign = ' ' + sign.strip() + ' '
  alternative_sign = ' ' + alternative_sign.strip() + ' '
  left_side_of_sign_or_alt_sign_is_valid_col = ((string.split(sign, maxsplit=1)[0].strip() in valid_cols) or (string.split(alternative_sign, maxsplit=1)[0].strip() in valid_cols))
  if (sign in string or alternative_sign in string) and left_side_of_sign_or_alt_sign_is_valid_col:
    string = replace_first_occurence_of_sign(string, sign, alternative_sign)
    col = string.split(alternative_sign.strip())[0].strip()
    value = string.split(alternative_sign.strip())[1].strip()
    if sign == ' is in ' or sign == ' is not in ':
      assert(value[0] == '[' and value[-1] == ']')
      value = [option.strip().strip('"').strip("'") for option in value[1:-1].split('|')] # handle the is-in strings like ['a', 'b', 'c'] or ["d", "e"]
    else:
      value = float(value) if value.isnumeric() else value
    return build_criteria(col, value, data, sign=sign)
  else:
    return None

def build_criteria_from_string(string, data):

  valid_cols = data.columns.tolist()

  criteria = check_for_criteria_type(string, data, ' is not in ', ' is not in ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria

  criteria = check_for_criteria_type(string, data, ' is not ', ' != ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria

  criteria = check_for_criteria_type(string, data, ' is in ', ' is in ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria

  criteria = check_for_criteria_type(string, data, ' is ', ' = ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria


def build_criteria(col, value, data, sign=' is '):

  if sign == ' is ' or sign == ' is not ':
    if value == 'MISSING':
      output = data[col].isnull()
    else:
      output = data[col] == value

  elif sign == ' is in ' or sign == ' is not in ':

    if pd.api.types.is_numeric_dtype(data[col].dtype) and len(value) == 2:
      output = (data[col] >= float(value[0])) & (data[col] < float(value[1]))
    else:
      output = data[col].isin(value)

  if ' not ' in sign:
    output = ~output

  return output

def get_multiple_criteria(string, data):
  if ' is in ' or 'is not in' in string:
    string = check_parenthesis_and_replace_comma_within_parenthesis(string)
  multiple_criteria = [c.strip() for c in string.split(',') if c.strip()!='']
  multiple_criteria = [c + ' = 1' if (' = ' not in c) and (' is ' not in c) and (c in data.columns) else c for c in multiple_criteria]
  multiple_criteria_filters = [build_criteria_from_string(c, data) for c in multiple_criteria]
  combined_filter = pd.Series([True] * len(data))
  for filter in multiple_criteria_filters:
    combined_filter = combined_filter & filter
  return combined_filter


def select_data(criteria_string, data):
  '''Provide a comma separated criteria_string, and specify which dataframe (df) to select from'''
  data = data.reset_index(drop=True).copy()
  criteria_filter = get_multiple_criteria(criteria_string, data)
  return data[criteria_filter].copy()

# ----------------------------------- Value Selection Based on Keywords ------------------------------------

def get_values_that_covers_threshold_percentage(col, data, thres = 0.9, order = 'most_first' ):

  ser = data.copy()[col].value_counts()
  cumsum_ser = (ser/ser.sum()).cumsum()
  top_values = cumsum_ser[cumsum_ser<=thres].index.tolist()
  if order == 'most_first':
    return top_values
  elif order == 'alphabetical':
    return sorted(top_values)
  else:
    return None


def check_contain(value, contain_list):
  contain_or_not = True
  for keyword in contain_list:
    if keyword not in value:
      contain_or_not = False
      break
  return contain_or_not

def check_not_contain(value, not_contain_list):
  not_contain_or_not = True
  for keyword in not_contain_list:
    if keyword in value:
      not_contain_or_not = False
      break
  return not_contain_or_not

def filter_by_keyword(input_list, contain = '', not_contain = '', case_important = False):

  output_list = []

  if not case_important:
    contain = contain.lower()
    not_contain = not_contain.lower()

  contain = [x.strip() for x in contain.split(',') if x!='']
  not_contain = [x.strip() for x in not_contain.split(',') if x!='']

  for value in input_list:
    if not case_important:
      processed_value = value.lower()
    else:
      processed_value = value

    if check_contain(processed_value, contain) and check_not_contain(processed_value, not_contain):
      output_list.append(value)

  return output_list

def filter_values(data, col, contain = '', not_contain = '' , coverage = 'auto', case_important = False, return_list = True, order = 'most_first'):
  if coverage == 'auto':
    if data[col].nunique()>100:
      input_list = get_values_that_covers_threshold_percentage(col, data, order = order)
    else:
      input_list = get_values_that_covers_threshold_percentage(col, data, thres = 1.0, order = order)
  elif coverage == 'full':
    input_list = get_values_that_covers_threshold_percentage(col, data, thres = 1.0, order = order)

  input_list = [item for item in input_list if isinstance(item,str)]

  output_list = filter_by_keyword(input_list, contain = contain, not_contain = not_contain, case_important = case_important)

  if return_list:
    return output_list
  else:
    print_list(output_list)

def show_filter_values(data, col, contain = '', not_contain = '' , coverage = 'auto', case_important = False, return_list = False, order = 'most_first'):
  return filter_values(data = data, col = col, contain = contain, not_contain = not_contain , coverage = coverage, case_important = case_important, return_list = return_list, order = order)

def change_values(data, orig_col, change_from, change_to, new_col = ''):
  if change_to == 'MISSING':
    change_to = np.nan
  if isinstance(change_from,list):
    pass
  else:
    change_from = [x.strip() for x in change_from.replace('\n','').split(',')  if x.strip()!='']
  if new_col == '':
    print(f'The column "{orig_col}" is changed.')
    new_col = orig_col
  else:
    print(f'A new column "{new_col}" is created to reflect the change.')
  data[new_col] = data[orig_col].apply(lambda x: change_to if x in change_from else x)

def filter_and_change_values(data, orig_col, contain = '', not_contain = '', change_to = None, new_col = '', coverage = 'auto', case_important = False):
  if change_to == None:
    print('Parameter "change_to" is not specificed.')
    return
  change_from = filter_values(data, orig_col,  contain = contain, not_contain = not_contain, coverage = coverage, case_important = case_important)
  print('Changing from:',)
  print_list(change_from,indent = 2)
  print('\nTo:\n  '+change_to+'\n')
  change_values(data, orig_col, change_from, change_to)
  return

# -------------------------------------Smart Data Description-------------------------------------------

def describe(col, data, top_k=-1, thres=90, return_full=False, plot_top_k=-1, plot_type='', bins=-1, show_graph = True):

  if data[col].isnull().mean() > 0:
    print(f"Notice: {np.round(data[col].isnull().mean()*100,3)}% of the entries have no records for this field.\n")

  data_numeric_columns = data.dtypes[data.dtypes.apply(lambda x: np.issubdtype(x, np.number))].index.tolist()

  if col in data_numeric_columns:
    if bins == -1:
      print(f'Change the default width of histogram bars by setting "bins = <a number>".\n')
      bins = 50
    plt.figure(figsize=(9, 6), dpi=default_dpi)
    plt.hist(data[col].dropna(), bins=bins)
    plt.title(f"Distribution of the {col}")
    basic_stats = data[col].dropna().describe().reset_index()
    basic_stats.columns = ['Field', 'Value']
    basic_stats.Field = ['Total Count', 'Mean', 'Standard Deviation', 'Minimum', 'Value at 25% Percentile', 'Median (50% Percentile)', 'Value at 75% Percentile', 'Maximum']
    basic_stats.loc[basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'] = basic_stats.loc[basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'].apply(lambda x: np.round(x, 2))
    basic_stats.loc[~basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'] = basic_stats.loc[~basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'].apply(int).apply(str)
    return basic_stats

  ser = data[col].value_counts()
  ser.name = 'Absolute Number'

  percentage_ser = np.round(ser / len(data) * 100, 2)
  percentage_ser.name = 'Proportion in Data (%)'

  cumsum_percentage_ser = percentage_ser.cumsum()
  cumsum_percentage_ser.name = 'Cumulative Proportion (%)'

  full_value_counts_df = pd.concat([ser, percentage_ser, cumsum_percentage_ser], axis=1)

  if plot_top_k > top_k:
    top_k = plot_top_k

  if top_k == -1:
    top_k = sum(cumsum_percentage_ser <= thres) + 1
    top_k = max(5, top_k)
    top_k = min(20, top_k)

  value_counts_df = full_value_counts_df if return_full else full_value_counts_df[:top_k]

  if top_k < len(full_value_counts_df) and not return_full:
    print(f'{len(full_value_counts_df)-top_k} more rows are available, add "return_full = True" if you want to see all.\n')

  if show_graph:
    plot_top_k = 10 if plot_top_k == -1 else plot_top_k
    graph_df = value_counts_df['Proportion in Data (%)'][:plot_top_k].copy()

    if plot_type == '':
      plot_type = 'bar' if graph_df.sum() < thres else 'pie'

    if plot_type == 'pie':

      fig, ax = plt.subplots(figsize=(9, 6), dpi=default_dpi, subplot_kw=dict(aspect="equal"))

      values = graph_df.values.tolist()
      names = graph_df.index.tolist()

      def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)

      wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values), textprops=dict(color="w"))

      for w in wedges:
        w.set_edgecolor('white')

      ax.legend(wedges, names,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.8, 1))

      plt.setp(autotexts, size=12, weight="bold")

      ax.set_title(f"Relative Proportion of Top {len(graph_df)} {col}" if len(graph_df) < len(full_value_counts_df) else f"Proportion of {col}")

    if plot_type == 'bar':
      plt.figure(figsize=(9, 6), dpi=default_dpi)
      graph_df.plot(kind='bar')
      plt.title(f"Barplot of the Top {len(graph_df)} {col} - (y axis shows percentage)")

    print()

  return value_counts_df


# # -----------------------------Correlation & Regression------------------------------------------

def show_corr(cols, data):

  if isinstance(cols, str):
    cols = [col.strip() for col in cols.strip().split(',')]
  try:
    corr_df = data[cols].copy().corr()
  except KeyError as e:
    print('Variable "' + re.findall(r"\[\'(.*?)\'\]", str(e))[0] + '" not found, check your spelling please.')
    return

  cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red green

  corr = corr_df.values
  np.fill_diagonal(corr, np.nan)
  corr = np.triu(corr, k=1)
  corr[corr == 0] = np.nan
  labels = corr_df.columns

  plt.figure(figsize=(5, 4), dpi=default_dpi)
  sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, xticklabels=labels, yticklabels=labels)


# def run_regression(design, data=regression_df):
#   variables = design.replace('~', ' ').replace('+', ' ').split()
#   selected_data = data[variables].copy()
#   scaled_data = pd.DataFrame(sklearn.preprocessing.RobustScaler().fit_transform(X=selected_data), columns=selected_data.columns)
#   y, X = patsy.dmatrices(design, data=scaled_data, return_type='dataframe')   # Split data columns
#   mod = sm.OLS(y, X)
#   res = mod.fit()
#   print(res.summary())


# -------------------- utilities ----------------------

def proportion(small_data, big_data, rounding = 3):
  return round(len(small_data)/len(big_data), rounding)

def print_list(list_to_print, indent=0, line_width=90):
  line_length = 0
  print(' ' * indent, end='')
  for val in list_to_print:
    val = str(val)
    line_length += len(val)
    print(val, end=', ')
    if line_length > line_width:
      print()
      print(' ' * indent, end='')
      line_length = 0
  print()

def save_graph(filename = '', quality = 'HD', padding = 0.3, transparent = False):

  orig_filename = filename

  if not os.path.exists('saved_graphs'):
    os.mkdir('saved_graphs')

  if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
    filename = filename+'.png'

  if filename == '':
    image_paths = [p.split('/')[-1].split('.')[0] for p in glob.glob('./saved_graphs/*')]
    available_indices = [int(p) for p in image_paths if p.isnumeric()]
    if len(available_indices) == 0:
      next_index = 1
    else:
      next_index = max(available_indices)+1
    filename = str(next_index).zfill(2)+'.png'

  if quality == 'SD':
    dpi = 90
  elif quality == 'HD':
    dpi = 150
  elif quality == 'Best':
    dpi = 300

  if orig_filename!='':
    plt.suptitle(orig_filename)

  plt.savefig('./saved_graphs/'+filename, dpi=dpi, bbox_inches='tight', transparent=transparent, pad_inches=padding)

