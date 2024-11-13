#!/bin/bash

mogrify -format jpg -crop 1000x450+100+20 *.png
