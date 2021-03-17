import React, { useState, useCallback, useEffect } from 'react'
import { Box, ChakraProvider, Button, Stack, Text, Heading, Grid, GridItem } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'



const App = () => {
  const [isSending, setIsSending] = useState(false)
  const [inferredImage, setInferredImage] = useState([])
  const [inferredLabel, setInferredLabel] = useState([])
  const [svg, setSvg] = useState('')

  const [
    divRef,
    {
      getSvgXML,
      undo,
      clear
    }
  ] = useSvgDrawing({
    penWidth: 6,     // pen width
    penColor: '#000000', // pen color
    width: 300, // drawing area width
    height: 300 // drawing area height
  })

  useEffect(() => {
    fetch('/api_list').then(response => console.log(response.json()))
  }, [])

  async function setInference(svg) {
    // Check that there is visible data in the svg
    if (svg.length < 500) {
      setInferredImage([])
      setInferredLabel([])
      return
    }

    // Send to back end
    const response = await fetch('/find_images', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ "sketch": svg })
    })

    // Receive response
    if (response.ok) {
      const res = await response.json()
      let inferredImages = res["images_base64"]
      let inferredLabels = res["images_label"]
      let tempImage = ''
      for (let i = 0; i < inferredImages.length; i++) {
        tempImage = inferredImages[i].split('\'')[1]
        inferredImages[i] = <img src={`data:image/jpeg;base64,${tempImage}`} alt='inferred_image' />
        inferredLabels[i] = `Guess ${i + 1}: ${inferredLabels[i]}`
      }
      setInferredImage(inferredImages)
      setInferredLabel(inferredLabels)
    }
  }


  const sendRequest = useCallback(async (svg) => {
    // don't send again while we are sending
    if (isSending) return
    // update state
    setIsSending(true)

    // set images and labels
    setInference(svg)

    // once the request is sent, update state again
    setIsSending(false)

  }, [isSending]) // update the callback if the state changes

  useEffect(() => {
    sendRequest(svg)
  }, [sendRequest, svg])

  return <ChakraProvider >
    <Stack
      spacing="0px"
      align="center">
      <Heading fontSize="4xl" color="teal">
        Sketchy App
      </Heading>
      <Grid
        h="95vh"
        w="99vw"
        templateRows="repeat(12, 1fr)"
        templateColumns="repeat(6, 1fr)"
        gap={4}
        align="center"
      >
        <GridItem rowSpan={1} colSpan={4}  >
          <Text fontSize="4xl" color="teal">
            Draw Sketch Here:
          </Text>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2}  >
          <Text fontSize="4xl" color="teal">
            Inferred Images:
          </Text>
        </GridItem>

        <GridItem rowSpan={9} colSpan={4} >
          <Box h="70vh" w="60vw" bg="#d0d5d9" borderRadius="md" ref={divRef}
            // onTouchEnd={() => sendRequest(getSvgXML())} // touch screen
            onMouseMove={() => setSvg(getSvgXML())}
          >
          </Box>
        </GridItem>
        <GridItem rowSpan={9} colSpan={2}  >
          <Box
            h="70vh"
            w="30vw"
            bg="#d0d5d9"
            borderRadius="md">
            <Text fontSize="2xl" color="teal">
              {inferredLabel[0]}
            </Text>
            {inferredImage[0]}

            <Box bg="#d0d5d9" w="90%" p={4} color="teal">
              ----------------------
            </Box>

            <Text fontSize="2xl" color="teal">
              {inferredLabel[1]}
            </Text>
            {inferredImage[1]}
          </Box>
        </GridItem>
        <GridItem rowSpan={2} colSpan={4}>
          <Button colorScheme="teal" size="lg" height="48px" width="180px" onClick={() => {
            undo()
            sendRequest(getSvgXML())
          }
          }>
            Undo last line
        </Button>
        </GridItem>
        <GridItem rowSpan={2} colSpan={2} color='blue' >
          <Button colorScheme="teal" size="lg" height="48px" width="180px" onClick={() => {
            clear()
            setInferredImage([])
            setInferredLabel([])
          }
          }>
            Restart!
          </Button>
        </GridItem>

      </Grid>
    </Stack>

  </ChakraProvider >
}

export default App
