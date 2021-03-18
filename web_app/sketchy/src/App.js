import React, { useState, useCallback, useEffect } from 'react'
import { Box, ChakraProvider, Button, Stack, Text, Heading, Grid, GridItem, CircularProgress } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'

const grey = "#d0d5d9"
const buttonColor = "teal"
const buttonHeight = "48px"
const buttonWidth = "180px"

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

  async function setInference(svg) {
    // Check that there is visible data in the svg
    if (svg.length < 500) {
      setInferredImage([])
      setInferredLabel([])
      return
    }

    // Show that we are processing the request
    setInferredImage([<CircularProgress isIndeterminate color="green.300" />,
    <CircularProgress isIndeterminate color="green.300" />])
    setInferredLabel(['Guess 1: ???', 'Guess 2: ???'])

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
    <Heading fontSize="4xl" color={buttonColor}>
      Sketchy App
    </Heading>
    <Stack
      spacing="0px"
      align="center">
      <Grid
        h="95vh"
        w="95vw"
        templateRows="repeat(12, 1fr)"
        templateColumns="repeat(6, 1fr)"
        gap={4}
        align="center"
      >
        <GridItem rowSpan={1} colSpan={4}  >
          <Text fontSize="4xl" color={buttonColor}>
            Draw Sketch Here:
          </Text>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2}  >
          <Text fontSize="4xl" color={buttonColor}>
            Inferred Images:
          </Text>
        </GridItem>
        <GridItem rowSpan={9} colSpan={4} >
          <Box h="70vh" w="62vw" bg={grey} borderRadius="md" ref={divRef}
            // onTouchEnd={() => sendRequest(getSvgXML())} // touch screen
            onMouseMove={() => {
              setSvg(getSvgXML())
            }}
          >
          </Box>
        </GridItem>
        <GridItem rowSpan={9} colSpan={2}  >
          <Box h="70vh" w="30vw" bg={grey} borderRadius="md">
            <Text fontSize="2xl" color={buttonColor}>
              {inferredLabel[0]}
            </Text>
            <Box bg={grey} w="90%" h="30%" p={0} color={buttonColor}>
              {inferredImage[0]}
            </Box>
            <Box bg={grey} w="90%" p={10} color={buttonColor} />
            <Text fontSize="2xl" color={buttonColor}>
              {inferredLabel[1]}
            </Text>
            <Box bg={grey} w="90%" h="30%" p={0} color={buttonColor}>
              {inferredImage[1]}
            </Box>
          </Box>
        </GridItem>
        <GridItem rowSpan={2} colSpan={4}>
          <Button colorScheme={buttonColor} size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
            undo()
            sendRequest(getSvgXML())
          }
          }>
            Undo last line
          </Button>
        </GridItem>
        <GridItem rowSpan={2} colSpan={2} >
          <Button colorScheme={buttonColor} size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
            clear()
            setInferredImage([])
            setInferredLabel([])
          }}>
            Restart!
          </Button>
        </GridItem>
      </Grid>
    </Stack>

  </ChakraProvider >
}

export default App
