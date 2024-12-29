def save_number_to_file(number, filename):
    try:
        with open(filename, 'a') as file:  # 'a'モードは追記モード
            if file.tell() > 0:  # ファイルが空でない場合、カンマを追加
                file.write(',')
            file.write(str(number))
        print(f"Number {number} has been added to {filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 例として数値42を保存
# number = 42
# filename = "number.txt"
# save_number_to_file(number, filename)
