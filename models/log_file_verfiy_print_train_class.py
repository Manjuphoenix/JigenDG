def process_log_file(file_path):
    class_loss_sum = 0.0
    class_loss_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            if 'class_loss' in line:
                # Extracting class_loss value
                class_loss = float(line.split('class_loss ')[1].split()[0])
                class_loss_sum += class_loss
                class_loss_count += 1

            # elif 'val jigsaw accuracy' in line or 'test jigsaw accuracy' in line:
            #     # Printing sum and average when encountering validation or test accuracy lines
            #     if class_loss_count > 0:
            #         average_class_loss = class_loss_sum / class_loss_count
            #         print(f"Sum of class_loss: {class_loss_sum}")
            #         print(f"Average of class_loss: {average_class_loss}")
            #     else:
            #         print("No class_loss found.")
            #     # Resetting counters for the next batch
            #     class_loss_sum = 0.0
            #     class_loss_count = 0

            elif 'val jigsaw' in line or 'test jigsaw' in line:
                # Printing sum and average when encountering validation or test accuracy lines
                if class_loss_count > 0:
                    average_class_loss = class_loss_sum / class_loss_count
                    # print(f"Sum of class_loss: {class_loss_sum}")
                    print(f"Average of class_loss: {average_class_loss}")
                else:
                    # print("No class_loss found.")
                    print("\n")
                # Resetting counters for the next batch
                class_loss_sum = 0.0
                class_loss_count = 0

# Usage
process_log_file('./JigenDG/outputs/jigen-da-change-ls-poly-00005-logging-updated-run2/trainandval.log')

