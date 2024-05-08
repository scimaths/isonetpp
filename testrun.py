import subprocess
import torch

# Define the arguments you want to test
arguments = ['1', '2']
dimin = ['1', '2', '3']
dimout = ['2', '3']
batch = ['1', '2', '3', '4']

for dimin1 in dimin:
    for dimout1 in dimout:
        for batch1 in batch:
            # Initialize variables to store the gradients
            grad_weight_diff = None
            grad_bias_diff = None

            # Iterate over arguments and run the script with each argument
            for arg in arguments:
                # Command to run the script with the current argument
                command = ['python', 'test.py', arg, dimin1, dimout1, batch1]
                
                # Execute the command
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for the process to finish and get the output
                stdout, stderr = process.communicate()
                
                # Parse the output to extract tensors
                print(stdout.decode())
                if stdout.decode() == '':
                    break
                output_lines = stdout.decode().split('weight\n')[1].replace('\n', '').split('bias')
                # Convert the string representations to tensors
                tensor_list = []
                for string in output_lines:
                    # Remove unwanted characters and split the string
                    string = string.replace('[', '').replace(']', '').split()
                    
                    # Convert the values into floats
                    float_list = [float(val) for val in string]
                    
                    # Create a tensor from the float list
                    tensor_list.append(torch.tensor(float_list))
                
                grad_weight = tensor_list[0]
                grad_bias = tensor_list[1]
                
                # Subtract gradients if both are available
                if grad_weight_diff is None:
                    grad_weight_diff = grad_weight
                    grad_bias_diff = grad_bias
                else:
                    grad_weight_diff -= grad_weight
                    grad_bias_diff -= grad_bias

            # Print the difference
            print("dimin:", dimin1, "dimout:", dimout1, "batch:", batch1)
            print("Difference in gradients (weight):", torch.sum(grad_weight_diff).item())
            print("Difference in gradients (bias):", torch.sum(grad_bias_diff).item())
