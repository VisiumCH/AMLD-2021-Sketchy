import React, { useEffect } from 'react'
import { Box, ChakraProvider, Button, Stack, Heading, HStack, StackDivider } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'



const App = () => {
  const [
    divRef,
    {
      getSvgXML,
      download,
      undo,
      clear
    }
  ] = useSvgDrawing({
    penWidth: 8,     // pen width
    penColor: '#000000', // pen color
    width: 300, // drawing area width
    height: 300 // drawing area height
  })

  useEffect(() => {
    fetch('/api_list').then(response => console.log(response.json()))
  }, [])


  // function SendSketch() {
  //   const svgDrawing = getSvgXML()
  //   console.log(getSvgXML())
  // useEffect(() => {
  //   fetch('/find_images').then(response => console.log(response.json()))
  // }, [])
  // }

  return <ChakraProvider >
    <Stack
      spacing={4}
      align="center">
      <Heading fontSize="5xl" color="teal">
        Draw Here:
      </Heading>
      <Box
        h="75vh"
        w="90vw"
        bg="#d0d5d9"
        borderRadius="md"
        ref={divRef}
      />

      <HStack
        divider={<StackDivider borderColor="gray.200" />}
        spacing={4}
        align="stretch"
      >
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          undo()
        }>
          Undo last line
        </Button>
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          clear()
        }>
          Erase everything
        </Button>
      </HStack>

      <HStack
        divider={<StackDivider borderColor="gray.200" />}
        spacing={4}
        align="stretch"
      >
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={async () => {
          const response = await fetch('/find_images', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "sketch": getSvgXML() })
          })
          if (response.ok) {
            console.log('response worked')
            console.log(response.json())
          } else {
            console.log('response did not worked')
          }
        }}>
          Send Sketch
        </Button>
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          download('png')
        }>
          Download Sketch
        </Button>
      </HStack>
    </Stack>
  </ChakraProvider >
}

export default App
