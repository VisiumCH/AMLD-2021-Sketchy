import { Box, ChakraProvider, Button, Stack, Heading, useCallback } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'



const App = () => {
  const [renderRef, draw] = useSvgDrawing({
    penWidth: 10,     // pen width
    penColor: '#000000', // pen color
    width: 300, // drawing area width
    height: 300 // drawing area height
  })

  return <ChakraProvider >
    <Stack align="center">
      <Heading fontSize="5xl" color="tomato">
        Draw Here:
      </Heading>
      <Box
        h="75vh"
        w="90vw"
        bg="#d0d5d9"
        borderRadius="md"
        ref={renderRef}
      />

      <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
        draw.clear()
      }>
        Erase all
    </Button>
      <Button colorScheme="teal" size="lg" height="48px" width="160px">
        Send
    </Button>
    </Stack>
  </ChakraProvider>
}

export default App
