import React, { useState, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { Box, ChakraProvider, Button, Text, Heading, VStack } from '@chakra-ui/react'

const darkGray = "#A3A8B0"
const textColor = "#FFFFFF"
const backgroundColor = "#1A365D"
const buttonHeight = "48px"
const buttonWidth = "180px"


function Embeddings() {
    const [isSending, setIsSending] = useState(false)
    const [x, setX] = useState([1, 2, 3])
    const [y, setY] = useState([4, 5, 6])
    const [z, setZ] = useState([7, 8, 9])
    const [legend, setLegend] = useState('Random Point')
    const [result, setResult] = useState({})
    let traces = []


    // const sendRequest = useCallback(async () => {
    //     // don't send again while we are sending
    //     if (isSending) return
    //     // update state
    //     setIsSending(true)

    //     // set images and labels
    //     // Send to back end
    //     const response = await fetch('/get_embeddings', {
    //         method: 'POST',
    //         headers: {
    //             'Content-Type': 'application/json'
    //         },
    //         body: JSON.stringify({ "body": "body" })
    //     })

    //     // Receive response
    //     if (response.ok) {
    //         const res = await response.json()
    //         console.log(res)

    //         for (let key in res) {
    //             let class_res = res[key]
    //             console.log(key)
    //             console.log(class_res)
    //     setX(class_res["x"])
    //     setY(class_res["y"])
    //     setZ(class_res["z"])
    //     setLegend(key)

    //     console.log(x)
    //     console.log(legend)
    //     // console.log(y)
    //     // console.log(z)
    //     let trace = Array(3).fill(0).map((_, i) => {
    //         return {
    //             x: x,
    //             y: y,
    //             z: z,
    //             name: legend,
    //             type: 'scatter3d',
    //             mode: 'markers',
    //             marker: {
    //                 color: 'red',
    //                 size: 4
    //             },
    //         }
    //     })
    //     traces.push(trace)
    // }
    // }

    //     // once the request is sent, update state again
    //     setIsSending(false)
    // }, [isSending])

    async function getEmbeddings() {
        // Send to back end
        const response = await fetch('/get_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "body": "body" })
        })

        if (response.ok) {
            const res = await response.json()
            console.log(res)
            for (let key in res) {
                console.log(key)
                console.log(res[key]['x'])
            }
        }

    }




    const sendRequest = useCallback(async (svg) => {
        // don't send again while we are sending
        if (isSending) return
        // update state
        setIsSending(true)

        // set images and labels
        getEmbeddings()

        // once the request is sent, update state again
        setIsSending(false)

    }, [isSending]) // update the callback if the state changes



    useEffect(() => {
        sendRequest()
    }, [sendRequest])

    // function isEmptyObject(obj) {
    //     return JSON.stringify(obj) === '{}';
    // }

    // if (isEmptyObject(res)) {
    //     console.log('Empty')
    //     console.log(res)
    // } else {

    //     console.log('not empty')
    //     console.log(res)
    // }


    // let traces = Array(3).fill(0).map((_, i) => {
    //     return {
    //         x: x,
    //         y: y,
    //         z: z,
    //         name: legend,
    //         type: 'scatter3d',
    //         mode: 'markers',
    //         marker: {
    //             color: 'red',
    //             size: 4
    //         },
    //     }
    // })

    return (
        <ChakraProvider >
            <Box bg={backgroundColor}>
                <VStack
                    spacing={4}
                    align="center"
                >
                    <Heading fontSize="4xl" color={textColor} align="center">
                        AMLD 2021 Visium's Sketchy App
                </Heading>
                    <Text fontSize="xs" color={textColor} align="center">
                        --------------------------------------------------------
                </Text>
                    <Text fontSize="xl" color={textColor} align="center">
                        Embeddings: Images and Sketches in latent space
                </Text>
                    <Plot
                        data={traces}
                        layout={{
                            width: 1200,
                            height: 650,
                            showlegend: true
                        }}
                    />
                    <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
                        sendRequest()
                    }}>
                        Load Graph
                    </Button>
                    <Link to="/drawing" className="drawing_link">
                        <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg"> Back to Drawing</Button>
                    </Link>
                    <Text fontSize="xs" color={textColor} align="center">

                    </Text>
                </VStack>
            </Box>
        </ChakraProvider >
    )
}

export default Embeddings