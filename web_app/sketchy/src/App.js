import React, { useState, useCallback, useEffect } from 'react'
import { Box, ChakraProvider, Button, Stack, HStack, VStack, Text, Heading, Grid, GridItem, CircularProgress } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'

const grey = "#d0d5d9"
const buttonColor = "teal"
const buttonHeight = "48px"
const buttonWidth = "180px"

const App = () => {
  const [isSending, setIsSending] = useState(false)
  const [inferredImage, setInferredImage] = useState([])
  const [inferredLabel, setInferredLabel] = useState([])
  const [attention, setAttention] = useState('')
  const [svg, setSvg] = useState('')

  const [
    divRef,
    {
      getSvgXML,
      undo,
      clear
    }
  ] = useSvgDrawing({
    penWidth: 3,     // pen width (similar as database width)
    penColor: '#000000', // pen color
    width: 300, // drawing area width
    height: 300 // drawing area height
  })

  async function setInference(svg) {
    // Check that there is visible data in the svg
    if (svg.length < 500) {
      setInferredImage([])
      setInferredLabel([])
      setAttention('')
      return
    }

    // Show that we are processing the request
    setInferredImage([<CircularProgress isIndeterminate color="green.300" />,
    <CircularProgress isIndeterminate color="green.300" />])
    setInferredLabel(['Guess 1: ???', 'Guess 2: ???'])
    setAttention(<CircularProgress isIndeterminate color="green.300" />)

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

      let tempAttention = res["attention"].split('\'')[1]
      setAttention(<img src={`data:image/jpeg;base64,${tempAttention}`} alt='attention_image' />)
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
    <Heading fontSize="4xl" color={buttonColor} align="center">
      Sketchy App
    </Heading>
    <Stack
      spacing="0px"
      align="center">
      <Grid
        h="95vh"
        w="95vw"
        templateRows="repeat(14, 1fr)"
        templateColumns="repeat(6, 1fr)"
        gap={2}
        align="center">
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
        <GridItem rowSpan={11} colSpan={4} >
          <Box h="70vh" w="62vw" bg={grey} borderRadius="md" ref={divRef}
            // onTouchEnd={() => sendRequest(getSvgXML())} // touch screen
            onMouseMove={() => {
              setSvg(getSvgXML())
            }}>
          </Box>
        </GridItem>


        <GridItem rowSpan={5} colSpan={2}  >

          <Box h="70vh" w="30vw" bg={grey} borderRadius="md">
            <HStack spacing="5px" align="center">
              <Box h="35vh" w="15vw" bg={grey} borderRadius="md">
                <VStack spacing="5px">
                  <Text fontSize="2xl" color={buttonColor}>
                    {inferredLabel[0]}
                  </Text>
                  <Box bg={grey} w="100%" h="35%" p={4} color={buttonColor}>
                    {inferredImage[0]}
                  </Box>
                </VStack>
              </Box>
              <Box h="35vh" w="15vw" bg={grey} borderRadius="md">
                <VStack spacing="5px">
                  <Text fontSize="2xl" color={buttonColor}>
                    {inferredLabel[1]}
                  </Text>
                  <Box bg={grey} w="100%" h="35%" p={4} color={buttonColor}>
                    {inferredImage[1]}
                  </Box>
                </VStack>
              </Box>
            </HStack>
            <Box bg={grey} w="90%" p={0} color={buttonColor} />
          </Box>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2} bg="white" >
          <Text fontSize="4xl" color={buttonColor}>
            Attention Map
          </Text>
        </GridItem>
        <GridItem rowSpan={5} colSpan={2}  >
          <Box bg={grey} w="37vh" color={buttonColor}>
            {attention}
          </Box>
        </GridItem>

        <GridItem rowSpan={2} colSpan={4}>
          <Button colorScheme={buttonColor} size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
            undo()
            sendRequest(getSvgXML())
          }}>
            Undo last line
          </Button>
        </GridItem>
        <GridItem rowSpan={2} colSpan={2} >
          <Button colorScheme={buttonColor} size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
            clear()
            setInferredImage([])
            setInferredLabel([])
            setAttention('')
          }}>
            Restart!
          </Button>
        </GridItem>
      </Grid>
    </Stack>
  </ChakraProvider >
}

export default App
