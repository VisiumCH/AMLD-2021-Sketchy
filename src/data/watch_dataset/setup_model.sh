OUTPUT_DIR='../../../io/models/watch_pretrained_models/hed'
mkdir -p $OUTPUT_DIR
wget https://github.com/ashukid/hed-edge-detector/blob/master/hed_pretrained_bsds.caffemodel?raw=true > $OUTPUT_DIR/hed_pretrained_bsds.caffemodel
curl https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt > $OUTPUT_DIR/deploy.prototxt
curl https://raw.githubusercontent.com/s9xie/hed/master/LICENSE > $OUTPUT_DIR/LICENSE