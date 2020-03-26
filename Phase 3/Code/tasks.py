import task1
import task2
import task3
import task4
import task5
import Task6_Main as task6
import preprocess


#Main Function with inf loop to select tasks
def main():

    while True :

        #Input task should be valid no
        try :
            task = int(input("Input the task no : "))
        except :
            #Error out
            print("Error : Invalid Task ... Exiting")
            return

        if task == 1:
            #Task 1 - Similar User using TF-DF-IDF
            task1.task()
        
        elif task == 0 :
            preprocess.img_img_sim_pickle()            
        
        elif task ==2:
            #Task 2 Similar Image using TF-DF-IDF
            task2.task()
        
        elif task == 3:
            #Task 2 Similar Image using TF-DF-IDF
            task3.task()
        elif task == 4:
            task4.task()
        elif task == 5:
            task5.task()
        elif task == 6:
            task6.task()
        else:
            break
        '''elif task ==2:
            #Task 2 Similar Image using TF-DF-IDF
            task2.task()
        
        elif task == 4:
            task4.task()
        elif task == 5:
            task5.task()
        elif task == 6:
            task6.task()
        elif task == 7:
            task7.task()
        '''
        '''
                elif task ==3:
                    #Task 3 Similar Location using TF-DF-IDF
                    task3.task()
        
                elif task ==4:
                    #Task 4 - Similar location using visual descriptor (1 out of 10)
                    task4.task()
        
                elif task ==5:
                    #Task 5 - Similar location using visual descriptor (all 10)
                    task5.task()
        '''
        

if __name__ == "__main__":
    main()
