/*************************************************************************
 * File Name: tensorRT_save_engine.cpp
 * Author: HaoXu 
 ************************************************************************/
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <memory>
#include <string.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} ;
static Logger gLogger;
void caffeToGIEModel(const char* deployFile, const char* modelFile, 
		const std::vector<std::string>& outputs, unsigned int maxBatchSize, 
		IHostMemory *&gieModelStream, std::string& serialize_str)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile, modelFile, *network, DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);

    std::cout<< "Begin building the engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout<< "End building the engine..." << std::endl;

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();

   //cppy serialize result from gieModelStream to serialize_str,and save it into "serialize_engine_output.txt".
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());   
    memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size());
    serialize_output_stream.open("./serialize_engine_output.txt");
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();

	// we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

int main()
{
    const char* prototxt = "./mnist.prototxt";
    const char* caffemodel = "./mnist.caffemodel";
    const char* OUTPUT_BLOB_NAME = "prob";
    IHostMemory *gieModelStream{nullptr};
    std::string engine_resialize_save;
    caffeToGIEModel(prototxt, caffemodel, std::vector < std::string > { OUTPUT_BLOB_NAME},1, gieModelStream,engine_resialize_save);
    return 0;   
}
