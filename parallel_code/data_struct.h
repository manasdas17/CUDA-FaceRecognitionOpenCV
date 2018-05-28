#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H

#include <cstddef>

struct Data
{
    double** images;
    unsigned int* labels;
    std::size_t size; // number of items
    std::size_t height; // height of each image
    std::size_t width;  // width of each image
    
    Data() : images(nullptr), labels(nullptr), size(0), width(0), height(0) {}
    
    Data(std::size_t size_) : images(new double*[size_]), labels(new unsigned int[size_]), size(size_), width(0), height(0) {}
    
    Data(std::size_t size_, std::size_t width_, std::size_t height_) :
        images(new double*[size_]), labels(new unsigned int[size_]), size(size_), width(width_), height(height_) {}
        
    Data( double** images_
        , unsigned int* labels_
        , std::size_t size_
        , std::size_t width_
        , std::size_t height_
        ) : images(images_), labels(labels_), size(size_), width(width_), height(height_) {}
        
    Data( Data&& rhs ) :
        images(rhs.images), labels(rhs.labels), size(rhs.size), width(rhs.width), height(rhs.height)
    { 
        for(std::size_t i = 0; i < size; ++i)
        {
            images[i] = rhs.images[i];
        }
        
        rhs.size = 0;
        rhs.width = 0;
        rhs.height = 0;
        rhs.images = nullptr;
        rhs.labels = nullptr;
    }
        
    ~Data()
    {
        if(size > 0)
        {
            for(std::size_t i = 0; i < size; ++i)
            {
                delete[] images[i];
            }
            
            delete[] images;
            delete[] labels;
        }
    }
    
    bool isEmpty() const
    {
        return size > 0;
    }
};

#endif //DATA_STRUCT_H
