#Q1> Create a NumPy array 'arr' of integers from 0 to 5 and print its data type.
import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5])
print(arr.dtype)


#Q2>Given a NumPy array 'arr', check if its data type is float64.

arr = np.array([1.5, 2.6, 3.7])

# Check if the data type is float64
if arr.dtype == np.float64:
    print("The data type is float64.")
else:
    print("The data type is not float64.")
    
    
#3. Create a NumPy array 'arr' with a data type of complex128 containing three complex numbers.
arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
print(arr)


#4. Convert an existing NumPy array 'arr' of integers to float32 data type.
arr = np.array([1, 2, 3, 4, 5])
arr_float32 = arr.astype(np.float32)
print(arr_float32)


#5. Given a NumPy array 'arr' with float64 data type, convert it to float32 to reduce decimal precision.
arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
arr_float32 = arr.astype(np.float32)
print(arr_float32)


#6. Write a function array_attributes that takes a NumPy array as input and returns its shape, size, and data
#type.
def array_attributes(arr):
    return arr.shape, arr.size, arr.dtype

# Example usage:
arr = np.array([[1, 2], [3, 4]])
print(array_attributes(arr))


#7. Create a function array_dimension that takes a NumPy array as input and returns its dimensionality.
def array_dimension(arr):
    return arr.ndim

# Example usage:
arr = np.array([[1, 2], [3, 4]])
print(array_dimension(arr))


#8. Design a function item_size_info that takes a NumPy array as input and returns the item size and the total
#size in bytes.
def item_size_info(arr):
    return arr.itemsize, arr.nbytes

# Example usage:
arr = np.array([1, 2, 3], dtype=np.int32)
print(item_size_info(arr))

#10. Design a function shape_stride_relationship that takes a NumPy array as input and returns the shape
#and strides of the array.
def shape_stride_relationship(arr):
    return arr.shape, arr.strides

# Example usage:
arr = np.array([[1, 2], [3, 4]])
print(shape_stride_relationship(arr))

#11. Create a function `create_zeros_array` that takes an integer `n` as input and returns a NumPy array of
#zeros with `n` elements.
def create_zeros_array(n):
    return np.zeros(n)

# Example usage:
print(create_zeros_array(5))

#12. Write a function `create_ones_matrix` that takes integers `rows` and `cols` as inputs and generates a 2D
#NumPy array filled with ones of size `rows x cols`.
def create_ones_matrix(rows, cols):
    return np.ones((rows, cols))

# Example usage:
print(create_ones_matrix(3, 2))


#13. Write a function `generate_range_array` that takes three integers start, stop, and step as arguments and
#creates a NumPy array with a range starting from `start`, ending at stop (exclusive), and with the specified
#`step`.
def generate_range_array(start, stop, step):
    return np.arange(start, stop, step)

# Example usage:
print(generate_range_array(0, 10, 2))

#14. Design a function `generate_linear_space` that takes two floats `start`, `stop`, and an integer `num` as
#arguments and generates a NumPy array with num equally spaced values between `start` and `stop`
#(inclusive).
def generate_linear_space(start, stop, num):
    return np.linspace(start, stop, num)

# Example usage:
print(generate_linear_space(0, 1, 5))

#15. Create a function `create_identity_matrix` that takes an integer `n` as input and generates a square
#identity matrix of size `n x n` using `numpy.eye`.
def create_identity_matrix(n):
    return np.eye(n)

# Example usage:
print(create_identity_matrix(3))


#16. Write a function that takes a Python list and converts it into a NumPy array.
def list_to_array(lst):
    return np.array(lst)

# Example usage:
print(list_to_array([1, 2, 3]))


#17. Create a NumPy array and demonstrate the use of `numpy.view` to create a new array object with the
#same data.
arr = np.array([1, 2, 3, 4])
arr_view = arr.view()
print("Original array:", arr)
print("View of the array:", arr_view)

# Modifying the view will affect the original array
arr_view[0] = 99
print("Modified view:", arr_view)
print("Original array after modification:", arr)

#18. Write a function that takes two NumPy arrays and concatenates them along a specified axis.


def concatenate_arrays(arr1, arr2, axis=0):
    return np.concatenate((arr1, arr2), axis=axis)

# Example usage:
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(concatenate_arrays(arr1, arr2, axis=0))


#19. Create two NumPy arrays with different shapes and concatenate them horizontally using `numpy.

#concatenate`.
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5], [6]])
result = np.concatenate((arr1, arr2), axis=1)
print(result)



#20. Write a function that vertically stacks multiple NumPy arrays given as a list.
def vertical_stack(arrays):
    return np.vstack(arrays)

# Example usage:
arr1 = np.array([1, 2])
arr2 = np.array([3, 4])
arr3 = np.array([5, 6])
print(vertical_stack([arr1, arr2, arr3]))


#21. Write a Python function using NumPy to create an array of integers within a specified range (inclusive)
#with a given step size.
def create_range_array(start, stop, step):
    return np.arange(start, stop + 1, step)

# Example usage:
print(create_range_array(1, 10, 2))


#22. Write a Python function using NumPy to generate an array of 10 equally spaced values between 0 and 1
#(inclusive).
def generate_equal_space():
    return np.linspace(0, 1, 10)

# Example usage:
print(generate_equal_space())


#23. Write a Python function using NumPy to create an array of 5 logarithmically spaced values between 1 and
#1000 (inclusive).
def generate_log_space():
    return np.logspace(0, 3, 5)  # log10(1) = 0, log10(1000) = 3

# Example usage:
print(generate_log_space())


#24. Create a Pandas DataFrame using a NumPy array that contains 5 rows and 3 columns, where the values
#are random integers between 1 and 100.
import pandas as pd

# Create a 5x3 NumPy array of random integers between 1 and 100
data = np.random.randint(1, 101, (5, 3))

# Create a Pandas DataFrame from the NumPy array
df = pd.DataFrame(data, columns=["Column1", "Column2", "Column3"])

print(df)


#25. Write a function that takes a Pandas DataFrame and replaces all negative values in a specific column
#with zeros. Use NumPy operations within the Pandas DataFrame.
def replace_negatives_with_zeros(df, column_name):
    df[column_name] = np.where(df[column_name] < 0, 0, df[column_name])
    return df

# Example usage:
data = {
    "A": [1, -2, 3, -4, 5],
    "B": [10, -20, 30, -40, 50]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Replace negative values in column "B" with zeros
df = replace_negatives_with_zeros(df, "B")
print("\nUpdated DataFrame:")
print(df)

#26.Access the 3rd element from the given NumPy array.


arr = np.array([10, 20, 30, 40, 50])

# Access the 3rd element (index 2 because indexing starts from 0)
third_element = arr[2]

print(third_element)

#27. Retrieve the element at index (1, 2) from the 2D NumPy array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Retrieve the element at index (1, 2)
element = arr_2d[1, 2]

print(element)


#28. Using boolean indexing, extract elements greater than 5 from the given NumPy array.

arr = np.array([3, 8, 2, 10, 5, 7])

# Use boolean indexing to extract elements greater than 5
elements_greater_than_5 = arr[arr > 5]

print(elements_greater_than_5)

#30. Slice the 2D NumPy array to extract the sub-array `[[2, 3], [5, 6]]` from the given array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Slice the array to extract the sub-array [[2, 3], [5, 6]]
sub_array = arr_2d[0:2, 1:3]

print(sub_array)


#33. Develop a NumPy function that extracts specific elements from a 3D array using indices provided in three
#separate arrays for each dimension.
def extract_elements(arr_2d, indices):
    return arr_2d[indices]

# Example usage:
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = (np.array([0, 1]), np.array([1, 2]))  # Extracting (0, 1) and (1, 2) elements
print(extract_elements(arr_2d, indices))

#32. Create a NumPy function that filters elements greater than a threshold from a given 1D array using
#boolean indexing.
def filter_elements(arr, threshold):
    return arr[arr > threshold]

# Example usage:
arr = np.array([1, 8, 3, 10, 5])
threshold = 5
print(filter_elements(arr, threshold))

#33. Develop a NumPy function that extracts specific elements from a 3D array using indices provided in three
#separate arrays for each dimension.
def extract_elements_3d(arr_3d, x_indices, y_indices, z_indices):
    return arr_3d[x_indices, y_indices, z_indices]

# Example usage:
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape (2, 2, 2)
x_indices = np.array([0, 1])
y_indices = np.array([0, 1])
z_indices = np.array([1, 0])

print(extract_elements_3d(arr_3d, x_indices, y_indices, z_indices))


#34. Write a NumPy function that returns elements from an array where both two conditions are satisfied
#using boolean indexing.
def extract_elements_2d(arr_2d, row_indices, col_indices):
    return arr_2d[row_indices, col_indices]

# Example usage:
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices = np.array([0, 2])
col_indices = np.array([1, 2])
print(extract_elements_2d(arr_2d, row_indices, col_indices))

#35. Create a NumPy function that extracts elements from a 2D array using row and column indices provided
#in separate arrays.
def extract_elements_2d(arr_2d, row_indices, col_indices):
    return arr_2d[row_indices, col_indices]

# Example usage:
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices = np.array([0, 2])
col_indices = np.array([1, 2])
print(extract_elements_2d(arr_2d, row_indices, col_indices))

#36. Given an array arr of shape (3, 3), add a scalar value of 5 to each element using NumPy broadcasting.
def add_scalar(arr, scalar):
    return arr + scalar

# Example usage:
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
scalar = 5
print(add_scalar(arr, scalar))


#37. Consider two arrays arr1 of shape (1, 3) and arr2 of shape (3, 4). Multiply each row of arr2 by the
#corresponding element in arr1 using NumPy broadcasting.
def multiply_rows(arr1, arr2):
    return arr2 * arr1.T  # Broadcasting arr1 to match the rows of arr2

# Example usage:
arr1 = np.array([[1, 3, 5]])  # Shape (1, 3)
arr2 = np.array([[2, 4, 6], [7, 8, 9], [10, 11, 12]])  # Shape (3, 3)
print(multiply_rows(arr1, arr2))


#38. Given a 1D array arr1 of shape (1, 4) and a 2D array arr2 of shape (4, 3), add arr1 to each row of arr2 using
#NumPy broadcasting.
def add_to_rows(arr1, arr2):
    return arr2 + arr1  # Broadcasting arr1 to each row of arr2

# Example usage:
arr1 = np.array([[1, 2, 3, 4]])  # Shape (1, 4)
arr2 = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]])  # Shape (4, 3)
print(add_to_rows(arr1, arr2))

#39. Consider two arrays arr1 of shape (3, 1) and arr2 of shape (1, 3). Add these arrays using NumPy
#broadcasting.
def add_arrays(arr1, arr2):
    return arr1 + arr2  # Broadcasting arr1 and arr2

# Example usage:
arr1 = np.array([[1], [2], [3]])  # Shape (3, 1)
arr2 = np.array([[4, 5, 6]])  # Shape (1, 3)
print(add_arrays(arr1, arr2))

#40. Given arrays arr1 of shape (2, 3) and arr2 of shape (2, 2), perform multiplication using NumPy
#broadcasting. Handle the shape incompatibility.
def multiply_arrays_broadcast(arr1, arr2):
    arr2_reshaped = arr2[:, np.newaxis]  # Reshape arr2 to make it compatible for broadcasting
    return arr1 * arr2_reshaped

# Example usage:
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
arr2 = np.array([[7, 8]])  # Shape (2, 2)
print(multiply_arrays_broadcast(arr1, arr2))


#41.Calculate column wise mean for the given array:
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Calculate column-wise mean
column_wise_mean = np.mean(arr, axis=0)

print(column_wise_mean)

#42. Find maximum value in each row of the given array:
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Find the maximum value in each row
row_wise_max = np.max(arr, axis=1)

print(row_wise_max)

#43. For the given array, find indices of maximum value in each column.
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Find indices of maximum value in each column
column_wise_max_indices = np.argmax(arr, axis=0)

print(column_wise_max_indices)

#44. For the given array, apply custom function to calculate moving sum along rows.
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Custom function to calculate moving sum along rows
def moving_sum(arr, window_size=2):
    result = np.apply_along_axis(lambda row: np.convolve(row, np.ones(window_size), mode='valid'), axis=1, arr=arr)
    return result

# Apply the moving sum function
moving_sum_result = moving_sum(arr, window_size=2)

print(moving_sum_result)

#45. In the given array, check if all elements in each column are even.
arr = np.array([[2, 4, 6], [3, 5, 7]])

# Check if all elements in each column are even
all_even_per_column = np.all(arr % 2 == 0, axis=0)

print(all_even_per_column)

#46. Given a NumPy array arr, reshape it into a matrix of dimensions `m` rows and `n` columns. Return the
#reshaped matrix.
original_array = np.array([1, 2, 3, 4, 5, 6])

m, n = 2, 3  
reshaped_matrix = original_array.reshape(m, n)

print(reshaped_matrix)

#47. Create a function that takes a matrix as input and returns the flattened array
def flatten_matrix(matrix):
    return matrix.flatten()

# Example usage:
input_matrix = np.array([[1, 2, 3], [4, 5, 6]])
flattened_array = flatten_matrix(input_matrix)
print(flattened_array)

#48.48. Write a function that concatenates two given arrays along a specified axis.
def concatenate_arrays(array1, array2, axis):
    return np.concatenate((array1, array2), axis=axis)

# Example usage:
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])
concatenated_array = concatenate_arrays(array1, array2, axis=0)  # Concatenate along rows (axis=0)
print(concatenated_array)

#49. Create a function that splits an array into multiple sub-arrays along a specified axis.
def split_array(array, axis, num_splits):
    return np.split(array, num_splits, axis=axis)

# Example usage:
original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
split_arrays = split_array(original_array, axis=0, num_splits=3)  # Split along rows
print(split_arrays)


#50. Write a function that inserts and then deletes elements from a given array at specified indices.
def insert_and_delete_elements(array, insert_indices, insert_values, delete_indices):
    array = np.insert(array, insert_indices, insert_values)
    array = np.delete(array, delete_indices)
    return array

# Example usage:
original_array = np.array([1, 2, 3, 4, 5])
indices_to_insert = [2, 4]
values_to_insert = [10, 11]
indices_to_delete = [1, 3]
modified_array = insert_and_delete_elements(original_array, indices_to_insert, values_to_insert, indices_to_delete)
print(modified_array)

#51. Create a NumPy array `arr1` with random integers and another array `arr2` with integers from 1 to 10.
#Perform element-wise addition between `arr1` and `arr2`.
arr1 = np.random.randint(1, 100, 5)  # Random integers between 1 and 100
arr2 = np.arange(1, 11, 2)  # Integers from 1 to 10 (odd numbers)

# Element-wise addition
result_add = arr1 + arr2

print("arr1:", arr1)
print("arr2:", arr2)
print("Element-wise addition result:", result_add)



#52. Generate a NumPy array `arr1` with sequential integers from 10 to 1 and another array `arr2` with integers
#from 1 to 10. Subtract `arr2` from `arr1` element-wise.
arr1 = np.arange(10, 0, -1)  # Sequential integers from 10 to 1
arr2 = np.arange(1, 11)  # Integers from 1 to 10

# Element-wise subtraction
result_subtract = arr1 - arr2

print("arr1:", arr1)
print("arr2:", arr2)
print("Element-wise subtraction result:", result_subtract)



#53. Create a NumPy array `arr1` with random integers and another array `arr2` with integers from 1 to 5.
#Perform element-wise multiplication between `arr1` and `arr2`.
arr1 = np.random.randint(1, 20, 5)  # Random integers between 1 and 20
arr2 = np.arange(1, 6)  # Integers from 1 to 5

# Element-wise multiplication
result_multiply = arr1 * arr2

print("arr1:", arr1)
print("arr2:", arr2)
print("Element-wise multiplication result:", result_multiply)


#54. Generate a NumPy array `arr1` with even integers from 2 to 10 and another array `arr2` with integers from 1
#to 5. Perform element-wise division of `arr1` by `arr2`.
arr1 = np.arange(2, 11, 2)  # Even integers from 2 to 10
arr2 = np.arange(1, 6)  # Integers from 1 to 5

# Element-wise division
result_divide = arr1 / arr2

print("arr1:", arr1)
print("arr2:", arr2)
print("Element-wise division result:", result_divide)


#55. Create a NumPy array `arr1` with integers from 1 to 5 and another array `arr2` with the same numbers
#reversed. Calculate the exponentiation of `arr1` raised to the power of `arr2` element-wise.
arr1 = np.arange(1, 6)  # Integers from 1 to 5
arr2 = np.flip(arr1)  # Reversed version of arr1

# Element-wise exponentiation
result_exponentiate = np.power(arr1, arr2)

print("arr1:", arr1)
print("arr2:", arr2)
print("Exponentiation result:", result_exponentiate)

#56. Write a function that counts the occurrences of a specific substring within a NumPy array of strings
def count_substring(arr, substring):
    return np.char.count(arr, substring).sum()

# Example usage:
arr = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
substring = 'hello'
count = count_substring(arr, substring)
print(count)

#57. Write a function that extracts uppercase characters from a NumPy array of strings.

arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])
def extract_uppercase(arr):
    return np.array([ ''.join([char for char in s if char.isupper()]) for s in arr])

# Example usage:
arr = np.array(['Hello', 'World', 'OpenAl', 'GPT'])
uppercase_chars = extract_uppercase(arr)
print(uppercase_chars)

#58. Write a function that replaces occurrences of a substring in a NumPy array of strings with a new string.
def replace_substring(arr, old_substring, new_substring):
    return np.char.replace(arr, old_substring, new_substring)

# Example usage:
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
old_substring = 'apple'
new_substring = 'orange'
modified_array = replace_substring(arr, old_substring, new_substring)
print(modified_array)

#59. Write a function that concatenates strings in a NumPy array element-wise.
def concatenate_strings(arr1, arr2):
    return np.core.defchararray.add(arr1, arr2)

# Example usage:
arr1 = np.array(['Hello', 'World'])
arr2 = np.array(['Open', 'AI'])
concatenated_array = concatenate_strings(arr1, arr2)
print(concatenated_array)

#60. Write a function that finds the length of the longest string in a NumPy array.

arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
def longest_string_length(arr):
    return max(np.char.str_len(arr))

# Example usage:
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
max_length = longest_string_length(arr)
print(max_length)


#61. Create a dataset of 100 random integers between 1 and 1000. Compute the mean, median, variance, and
#standard deviation of the dataset using NumPy's functions.
dataset = np.random.randint(1, 1001, 100)

# Compute mean, median, variance, and standard deviation
mean = np.mean(dataset)
median = np.median(dataset)
variance = np.var(dataset)
std_deviation = np.std(dataset)

print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Standard Deviation:", std_deviation)


#62. Generate an array of 50 random numbers between 1 and 100. Find the 25th and 75th percentiles of the
#dataset.
arr = np.random.randint(1, 101, 50)

# Find the 25th and 75th percentiles
percentile_25 = np.percentile(arr, 25)
percentile_75 = np.percentile(arr, 75)

print("25th Percentile:", percentile_25)
print("75th Percentile:", percentile_75)


#63. Create two arrays representing two sets of variables. Compute the correlation coefficient between these
#arrays using NumPy's `corrcoef` function.
# Create two matrices
matrix1 = np.random.randint(1, 10, (2, 3))  # 2x3 matrix
matrix2 = np.random.randint(1, 10, (3, 2))  # 3x2 matrix

# Matrix multiplication using dot
matrix_product = np.dot(matrix1, matrix2)

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Matrix Multiplication Result:\n", matrix_product)


#64. Create two matrices and perform matrix multiplication using NumPy's `dot` function.
# Create two matrices
matrix1 = np.random.randint(1, 10, (2, 3))  # 2x3 matrix
matrix2 = np.random.randint(1, 10, (3, 2))  # 3x2 matrix

# Matrix multiplication using dot
matrix_product = np.dot(matrix1, matrix2)

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Matrix Multiplication Result:\n", matrix_product)


#65. Create an array of 50 integers between 10 and 1000. Calculate the 10th, 50th (median), and 90th
#percentiles along with the first and third quartiles.
arr = np.random.randint(10, 1001, 50)

# Calculate the percentiles
percentile_10 = np.percentile(arr, 10)
percentile_50 = np.percentile(arr, 50)  # Median
percentile_90 = np.percentile(arr, 90)

# Calculate first and third quartiles
first_quartile = np.percentile(arr, 25)
third_quartile = np.percentile(arr, 75)

print("10th Percentile:", percentile_10)
print("50th Percentile (Median):", percentile_50)
print("90th Percentile:", percentile_90)
print("First Quartile:", first_quartile)
print("Third Quartile:", third_quartile)


#66. Create a NumPy array of integers and find the index of a specific element.
arr = np.random.randint(1, 100, 10)

# Find index of a specific element
element = 50  # Example element
index_of_element = np.where(arr == element)

print("Array:", arr)
print("Index of element", element, ":", index_of_element)


#67. Generate a random NumPy array and sort it in ascending order.
arr = np.random.randint(1, 100, 10)

# Sort the array in ascending order
sorted_arr = np.sort(arr)

print("Original Array:", arr)
print("Sorted Array:", sorted_arr)

#68. Filter elements >20  in the given NumPy array.
def filter_greater_than_20(arr):
    return arr[arr > 20]

# Example usage:
arr = np.array([12, 25, 6, 42, 8, 30])
filtered_arr = filter_greater_than_20(arr)
print(filtered_arr)

#69. Filter elements which are divisible by 3 from a given NumPy array.
def filter_divisible_by_3(arr):
    return arr[arr % 3 == 0]

# Example usage:
arr = np.array([1, 5, 8, 12, 15])
filtered_arr = filter_divisible_by_3(arr)
print(filtered_arr)

#70. Filter elements which are ≥ 20 and ≤ 40 from a given NumPy array.
def filter_range_20_to_40(arr):
    return arr[(arr >= 20) & (arr <= 40)]

# Example usage:
arr = np.array([10, 20, 30, 40, 50])
filtered_arr = filter_range_20_to_40(arr)
print(filtered_arr)


#71. For the given NumPy array, check its byte order using the `dtype` attribute byteorder.
def check_byte_order(arr):
    return arr.dtype.byteorder

# Example usage:
arr = np.array([1, 2, 3])
byte_order = check_byte_order(arr)
print(byte_order)

#76. Create a NumPy array `arr1` with values from 1 to 10. Create a copy of `arr1` named `copy_arr` and modify
#an element in `copy_arr`. Check if modifying `copy_arr` affects `arr1`.
arr1 = np.arange(1, 11)

# Create a copy of arr1
copy_arr = arr1.copy()

# Modify an element in copy_arr
copy_arr[0] = 100

print("Original Array (arr1):", arr1)
print("Modified Copy Array (copy_arr):", copy_arr)


#77. Create a 2D NumPy array `matrix` of shape (3, 3) with random integers. Extract a slice `view_slice` from
#the matrix. Modify an element in `view_slice` and observe if it changes the original `matrix`.
# Create matrix
matrix = np.random.randint(1, 10, (3, 3))

# Extract a slice
view_slice = matrix[:2, :2]

# Modify an element in view_slice
view_slice[0, 0] = 100

print("Original Matrix:\n", matrix)
print("Modified View Slice:\n", view_slice)


#78. Create a NumPy array `array_a` of shape (4, 3) with sequential integers from 1 to 12. Extract a slice
#`view_b` from `array_a` and broadcast the addition of 5 to view_b. Check if it alters the original `array_a`.
# Create array_a
array_a = np.arange(1, 13).reshape(4, 3)

# Extract a slice
view_b = array_a[:2, :2]

# Broadcast the addition of 5 to view_b
view_b += 5

print("Original Array (array_a):\n", array_a)
print("Modified View Slice (view_b):\n", view_b)


#79. Create a NumPy array `orig_array` of shape (2, 4) with values from 1 to 8. Create a reshaped view
#`reshaped_view` of shape (4, 2) from orig_array. Modify an element in `reshaped_view` and check if it
#reflects changes in the original `orig_array`.
# Create orig_array
orig_array = np.arange(1, 9).reshape(2, 4)

# Create reshaped view
reshaped_view = orig_array.T

# Modify an element in reshaped_view
reshaped_view[0, 0] = 100

print("Original Array (orig_array):\n", orig_array)
print("Modified Reshaped View (reshaped_view):\n", reshaped_view)


#80. Create a NumPy array `data` of shape (3, 4) with random integers. Extract a copy `data_copy` of
#elements greater than 5. Modify an element in `data_copy` and verify if it affects the original `data`.

data = np.random.randint(1, 10, (3, 4))

# Extract a copy of elements greater than 5
data_copy = data[data > 5].copy()

# Modify an element in data_copy
data_copy[0] = 100

print("Original Data Array:\n", data)
print("Modified Data Copy:\n", data_copy)


#81. Create two matrices A and B of identical shape containing integers and perform addition and subtraction
#operations between them.
# Create matrices A and B
A = np.random.randint(1, 10, (3, 3))
B = np.random.randint(1, 10, (3, 3))

# Perform addition and subtraction
add_result = A + B
sub_result = A - B

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix Addition (A + B):\n", add_result)
print("Matrix Subtraction (A - B):\n", sub_result)


#82. Generate two matrices `C` (3x2) and `D` (2x4) and perform matrix multiplication.
# Create matrices C and D
C = np.random.randint(1, 10, (3, 2))
D = np.random.randint(1, 10, (2, 4))

# Perform matrix multiplication
mul_result = np.dot(C, D)

print("Matrix C:\n", C)
print("Matrix D:\n", D)
print("Matrix Multiplication (C * D):\n", mul_result)


#83. Create a matrix `E` and find its transpose.
# Create matrix E
E = np.random.randint(1, 10, (2, 3))

# Find the transpose of matrix E
transpose_E = E.T

print("Matrix E:\n", E)
print("Transpose of Matrix E:\n", transpose_E)


#84. Generate a square matrix `F` and compute its determinant.
# Create a square matrix F
F = np.random.randint(1, 10, (3, 3))

# Compute the determinant of F
det_F = np.linalg.det(F)

print("Matrix F:\n", F)
print("Determinant of Matrix F:", det_F)


#85. Create a square matrix `G` and find its inverse.
G = np.random.randint(1, 10, (3, 3))

# Print the original matrix
print("Matrix G:\n", G)

# Find the inverse of matrix G
inv_G = np.linalg.inv(G)

# Print the inverse of the matrix
print("Inverse of Matrix G:\n", inv_G)
