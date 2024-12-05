
# Sierpinski fractal on GPU

The [Sierpiński triangle](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle) is 
a *fractal* with the overall shape of an equilateral triangle. There are two main
approaches for the construction of the fractal, and we will focus on the so-called 
["shrinking and duplication" procedure](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle#Shrinking_and_duplication).
Starting from any given image, the procedure, when iterated to infinity, will always
give as a result a Sierpiński triangle. In our finite-dimensional computers, we will
approximate the infinity with the latest iteration of the procedure for which there
will actually be changes on the pixels composing our images.

The provided [CUDA code](./sierpinski.cu) makes use of CUDA streams to launch several
kernels in parallel on the same GPU device. But why to launch more than one kernel 
per time? And can you see why this can be done without worrying about concurrency 
issues?

## Running the CUDA program

Execution on a Nvidia T4 Tesla GPU card. In our small black-and-white images (see 
setup in the CUDA code), the "infinity" is attained only after 6 iterations of the 
procedure, but the Sierpiński triangle is already recognizable.

	Sierpinski on GPU with CUDA
	Two-dimensional thread grid structures:
	* Main grid [blocks (16,16), threads (8,8)]
	* Shrunk grid [blocks (16,16), threads (4,4)]
	                                                               xx                                                               
	                                                               xx                                                               
	                                                              xxxx                                                              
	                                                              xxxx                                                              
	                                                             xx  xx                                                             
	                                                             xx  xx                                                             
	                                                            xxxxxxxx                                                            
	                                                            xxxxxxxx                                                            
	                                                           xx      xx                                                           
	                                                           xx      xx                                                           
	                                                          xxxx    xxxx                                                          
	                                                          xxxx    xxxx                                                          
	                                                         xx  xx  xx  xx                                                         
	                                                         xx  xx  xx  xx                                                         
	                                                        xxxxxxxxxxxxxxxx                                                        
	                                                        xxxxxxxxxxxxxxxx                                                        
	                                                       xx              xx                                                       
	                                                       xx              xx                                                       
	                                                      xxxx            xxxx                                                      
	                                                      xxxx            xxxx                                                      
	                                                     xx  xx          xx  xx                                                     
	                                                     xx  xx          xx  xx                                                     
	                                                    xxxxxxxx        xxxxxxxx                                                    
	                                                    xxxxxxxx        xxxxxxxx                                                    
	                                                   xx      xx      xx      xx                                                   
	                                                   xx      xx      xx      xx                                                   
	                                                  xxxx    xxxx    xxxx    xxxx                                                  
	                                                  xxxx    xxxx    xxxx    xxxx                                                  
	                                                 xx  xx  xx  xx  xx  xx  xx  xx                                                 
	                                                 xx  xx  xx  xx  xx  xx  xx  xx                                                 
	                                                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                                
	                                                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                                
	                                               xx                              xx                                               
	                                               xx                              xx                                               
	                                              xxxx                            xxxx                                              
	                                              xxxx                            xxxx                                              
	                                             xx  xx                          xx  xx                                             
	                                             xx  xx                          xx  xx                                             
	                                            xxxxxxxx                        xxxxxxxx                                            
	                                            xxxxxxxx                        xxxxxxxx                                            
	                                           xx      xx                      xx      xx                                           
	                                           xx      xx                      xx      xx                                           
	                                          xxxx    xxxx                    xxxx    xxxx                                          
	                                          xxxx    xxxx                    xxxx    xxxx                                          
	                                         xx  xx  xx  xx                  xx  xx  xx  xx                                         
	                                         xx  xx  xx  xx                  xx  xx  xx  xx                                         
	                                        xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx                                        
	                                        xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx                                        
	                                       xx              xx              xx              xx                                       
	                                       xx              xx              xx              xx                                       
	                                      xxxx            xxxx            xxxx            xxxx                                      
	                                      xxxx            xxxx            xxxx            xxxx                                      
	                                     xx  xx          xx  xx          xx  xx          xx  xx                                     
	                                     xx  xx          xx  xx          xx  xx          xx  xx                                     
	                                    xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx                                    
	                                    xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx                                    
	                                   xx      xx      xx      xx      xx      xx      xx      xx                                   
	                                   xx      xx      xx      xx      xx      xx      xx      xx                                   
	                                  xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx                                  
	                                  xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx                                  
	                                 xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx                                 
	                                 xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx                                 
	                                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                
	                                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                
	                               xx                                                              xx                               
	                               xx                                                              xx                               
	                              xxxx                                                            xxxx                              
	                              xxxx                                                            xxxx                              
	                             xx  xx                                                          xx  xx                             
	                             xx  xx                                                          xx  xx                             
	                            xxxxxxxx                                                        xxxxxxxx                            
	                            xxxxxxxx                                                        xxxxxxxx                            
	                           xx      xx                                                      xx      xx                           
	                           xx      xx                                                      xx      xx                           
	                          xxxx    xxxx                                                    xxxx    xxxx                          
	                          xxxx    xxxx                                                    xxxx    xxxx                          
	                         xx  xx  xx  xx                                                  xx  xx  xx  xx                         
	                         xx  xx  xx  xx                                                  xx  xx  xx  xx                         
	                        xxxxxxxxxxxxxxxx                                                xxxxxxxxxxxxxxxx                        
	                        xxxxxxxxxxxxxxxx                                                xxxxxxxxxxxxxxxx                        
	                       xx              xx                                              xx              xx                       
	                       xx              xx                                              xx              xx                       
	                      xxxx            xxxx                                            xxxx            xxxx                      
	                      xxxx            xxxx                                            xxxx            xxxx                      
	                     xx  xx          xx  xx                                          xx  xx          xx  xx                     
	                     xx  xx          xx  xx                                          xx  xx          xx  xx                     
	                    xxxxxxxx        xxxxxxxx                                        xxxxxxxx        xxxxxxxx                    
	                    xxxxxxxx        xxxxxxxx                                        xxxxxxxx        xxxxxxxx                    
	                   xx      xx      xx      xx                                      xx      xx      xx      xx                   
	                   xx      xx      xx      xx                                      xx      xx      xx      xx                   
	                  xxxx    xxxx    xxxx    xxxx                                    xxxx    xxxx    xxxx    xxxx                  
	                  xxxx    xxxx    xxxx    xxxx                                    xxxx    xxxx    xxxx    xxxx                  
	                 xx  xx  xx  xx  xx  xx  xx  xx                                  xx  xx  xx  xx  xx  xx  xx  xx                 
	                 xx  xx  xx  xx  xx  xx  xx  xx                                  xx  xx  xx  xx  xx  xx  xx  xx                 
	                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                
	                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                
	               xx                              xx                              xx                              xx               
	               xx                              xx                              xx                              xx               
	              xxxx                            xxxx                            xxxx                            xxxx              
	              xxxx                            xxxx                            xxxx                            xxxx              
	             xx  xx                          xx  xx                          xx  xx                          xx  xx             
	             xx  xx                          xx  xx                          xx  xx                          xx  xx             
	            xxxxxxxx                        xxxxxxxx                        xxxxxxxx                        xxxxxxxx            
	            xxxxxxxx                        xxxxxxxx                        xxxxxxxx                        xxxxxxxx            
	           xx      xx                      xx      xx                      xx      xx                      xx      xx           
	           xx      xx                      xx      xx                      xx      xx                      xx      xx           
	          xxxx    xxxx                    xxxx    xxxx                    xxxx    xxxx                    xxxx    xxxx          
	          xxxx    xxxx                    xxxx    xxxx                    xxxx    xxxx                    xxxx    xxxx          
	         xx  xx  xx  xx                  xx  xx  xx  xx                  xx  xx  xx  xx                  xx  xx  xx  xx         
	         xx  xx  xx  xx                  xx  xx  xx  xx                  xx  xx  xx  xx                  xx  xx  xx  xx         
	        xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx        
	        xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx                xxxxxxxxxxxxxxxx        
	       xx              xx              xx              xx              xx              xx              xx              xx       
	       xx              xx              xx              xx              xx              xx              xx              xx       
	      xxxx            xxxx            xxxx            xxxx            xxxx            xxxx            xxxx            xxxx      
	      xxxx            xxxx            xxxx            xxxx            xxxx            xxxx            xxxx            xxxx      
	     xx  xx          xx  xx          xx  xx          xx  xx          xx  xx          xx  xx          xx  xx          xx  xx     
	     xx  xx          xx  xx          xx  xx          xx  xx          xx  xx          xx  xx          xx  xx          xx  xx     
	    xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx    
	    xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx        xxxxxxxx    
	   xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx   
	   xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx      xx   
	  xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx  
	  xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx    xxxx  
	 xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx 
	 xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx  xx 
	xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Links

* [Back to HPC lectures](./README.md)
* [Back to main repository page](../README.md)

