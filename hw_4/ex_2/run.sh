#!/bin/bash
for i in {102400..512000..12800}
do
    ./lab4_ex2_nonstreamed $i    
done

for i in {102400..512000..12800}
do
    ./lab4_ex2 $i $(echo $(expr $i / 4))    
done

