import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import Plot from 'react-plotly.js';
import {
  Box,
  Text,
  Stack,
  VStack,
  Grid,
  GridItem,
  Radio,
  RadioGroup,
} from '@chakra-ui/react';
import { colors, custom_sketch_class } from '../constants';
import { heading } from '../ui_utils';
import { PageDrawer } from '../drawer.js';

function Embeddings() {
  const { state } = useLocation();
  const [result, setResult] = useState({});
  const [nbDimensions, setNbDimensions] = useState('3');
  const [reductionAlgo, setReductionAlgo] = useState('PCA');
  const [clickedClass, setClickedClass] = useState('');
  const [clickedX, setClickedX] = useState(0);
  const [clickedY, setClickedY] = useState(0);
  const [clickedZ, setClickedZ] = useState(0);
  const [sketch, setSketch] = useState([]);
  const [showImage, setShowImage] = useState([]);
  const point = useRef({ clickedX, clickedY, clickedZ });

  let traces = [];

  // get the embedding graph in 2D or 3D
  useEffect(() => {
    let to_send = { nb_dim: nbDimensions, reduction_algo: reductionAlgo };
    if (state !== 'undefined') {
      to_send['sketch'] = state;
      setClickedClass(custom_sketch_class);
    }

    async function getEmbeddings(to_send) {
      // Send to back end
      const response = await fetch('/api/get_embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(to_send),
      });

      if (response.ok) {
        const res = await response.json();
        setResult(res);
      }
    }

    getEmbeddings(to_send);
  }, [state, nbDimensions, reductionAlgo]);

  useEffect(() => {
    let to_send = {
      class: clickedClass,
      nb_dim: nbDimensions,
      reduction_algo: reductionAlgo,
    };
    if (clickedClass === custom_sketch_class) {
      to_send['sketch'] = state;
    } else {
      if (
        point.current.clickedX === clickedX ||
        point.current.clickedY === clickedY ||
        (nbDimensions === '3' && point.current.clickedZ === clickedZ)
      ) {
        return;
      } else {
        to_send['x'] = clickedX;
        to_send['y'] = clickedY;
        if (nbDimensions === '3') {
          to_send['z'] = clickedZ;
        }
        point.current = { clickedX, clickedY, clickedZ };
      }
    }

    async function getClickedImage(to_send) {
      // Send to back end
      const response = await fetch('/api/get_embedding_images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(to_send),
      });

      if (response.ok) {
        const res = await response.json();
        let tempImage = res['image'].split("'")[1];
        if (clickedClass === custom_sketch_class) {
          setSketch(
            <img
              src={`data:image/jpeg;base64,${tempImage}`}
              alt='inferred_image'
            />
          );
        } else {
          setShowImage(
            <img
              src={`data:image/jpeg;base64,${tempImage}`}
              alt='inferred_image'
            />
          );
        }
      }
    }
    getClickedImage(to_send);
  }, [clickedClass, clickedX, clickedY, clickedZ, state]);

  function fillTraces() {
    let marker_size = 6;
    let marker_line_color = colors[0];
    let marker_line_width = 0.1;
    let i = 0;
    for (let key in result) {
      if (key === custom_sketch_class) {
        marker_size = 10;
        marker_line_color = 'rgb(0,0,0)';
        marker_line_width = 1;
      } else {
        marker_line_color = colors[i];
        marker_line_width = 0.1;
        marker_size = 6;
      }

      let trace = {
        x: result[key]['x'],
        y: result[key]['y'],
        name: key,
        mode: 'markers',
        marker: {
          color: colors[i],
          line: {
            color: marker_line_color,
            width: marker_line_width,
          },
        },
        hoverinfo: 'name',
      };
      if (nbDimensions === '3') {
        trace['z'] = result[key]['z'];
        trace['type'] = 'scatter3d';
        trace['marker']['size'] = marker_size;
      } else {
        trace['type'] = 'scatter';
        trace['marker']['size'] = marker_size * 2;
      }
      traces.push(trace);
      i = i + 1;
    }
  }
  fillTraces();

  function showSketch() {
    if (state !== 'undefined') {
      return sketch;
    } else {
      return (
        <Text fontSize='xl' color='white' align='center'>
          No sketch drawn, go to "Draw" to draw one!
        </Text>
      );
    }
  }

  function getClickedImage(e) {
    setClickedClass(e['points'][0]['data']['name']);
    setClickedX(e['points'][0]['x']);
    setClickedY(e['points'][0]['y']);
    setClickedZ(e['points'][0]['z']);
  }

  return (
    <>
      {heading}
      <Text fontSize='2xl' color='white' align='center'>
        Embeddings: Images and Sketches in {nbDimensions}D after {reductionAlgo}{' '}
        projection
      </Text>
      <Box p='2' bg='darkBlue'>
        <Grid
          h='81.5vh'
          gap={4}
          align='center'
          templateRows='repeat(1, 1fr)'
          templateColumns='repeat(10, 1fr)'
        >
          <GridItem rowSpan={1} colSpan={2} align='center'>
            <VStack spacing={3} direction='row' align='center'>
              <Text fontSize='2xl' color='white' align='center' as='u'>
                Options
              </Text>
              <RadioGroup
                onChange={setNbDimensions}
                value={nbDimensions}
                colorScheme='white'
                defaultValue='2'
                color='white'
                size='lg'
              >
                <Stack direction='row'>
                  <Radio value='2'>2D</Radio>
                  <Radio value='3'>3D</Radio>
                </Stack>
              </RadioGroup>
              <RadioGroup
                onChange={setReductionAlgo}
                value={reductionAlgo}
                colorScheme='white'
                defaultValue='2'
                color='white'
                size='lg'
              >
                <Stack direction='row'>
                  <Radio value='PCA'>PCA</Radio>
                  <Radio value='TSNE'>TSNE</Radio>
                  <Radio value='UMAP'>UMAP</Radio>
                </Stack>
              </RadioGroup>
              <Text fontSize='2xl' color='white' align='center' as='u'>
                My Sketch
              </Text>
              {showSketch()}
              <Text fontSize='2xl' color='white' align='center' as='u'>
                Clicked image
              </Text>
              {showImage}
            </VStack>
          </GridItem>

          <GridItem rowSpan={1} colSpan={8}>
            <Plot
              data={traces}
              layout={{
                width: 1200,
                height: 680,
                hovermode: 'closest',
                showlegend: true,
                margin: {
                  l: 0,
                  r: 0,
                  b: 0,
                  t: 0,
                },
                legend: {
                  title: {
                    text: 'Categories',
                    font: {
                      size: 20,
                      color: 'darkBlue',
                    },
                  },
                  font: {
                    size: 16,
                    color: 'darkBlue',
                  },
                  orientation: 'v',
                  itemsizing: 'constant',
                  x: 0.9,
                  y: 0.5,
                },
                font: {
                  color: 'darkBlue',
                },
                paper_bgcolor: 'white',
              }}
              onClick={(e) => getClickedImage(e)}
            />
          </GridItem>
        </Grid>
        {PageDrawer()}
      </Box>
    </>
  );
}

export default Embeddings;
