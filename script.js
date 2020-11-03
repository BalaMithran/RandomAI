
// Preparing the training data 

function generaterandom(num) {
    var seq = []
    while (num > 0) {
        t = Math.floor(Math.random() * 100)
        seq.push(t)
        num--;
    }
    return seq;
}

const fibs = generaterandom(100)
const xs = tf.tensor1d(fibs.slice(0, fibs.length - 1))
const ys = tf.tensor1d(fibs.slice(1))
const xmin = xs.min();
const xmax = xs.max();
const xrange = xmax.sub(xmin);
function norm(x) {
    return x.sub(xmin).div(xrange);
}
xsNorm = norm(xs)
ysNorm = norm(ys)
// Building our model
const a = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))
a.print()
b.print()
function predict(x) {
    return tf.tidy(() => {
        return a.mul(x).add(b)
    });
}

function train() {
    // Training
    function loss(predictions, labels) {
        return predictions.sub(labels).square().mean();
    }
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);
    const numIterations = 10000;
    const errors = []
    for (let iter = 0; iter < numIterations; iter++) {
        optimizer.minimize(() => {
            const predsYs = predict(xsNorm);
            const e = loss(predsYs, ysNorm);
            errors.push(e.dataSync())
            return e
        });
    }
}




function reload() {

    // Making predictions

    // a.print()
    // b.print()

    let array = []
    while (array.length <= 100) {
        temp = Math.floor(Math.random() * 100)
        xTest = tf.tensor1d([temp])
        // predict(xTest).print()
        let prediction = predict(xTest).dataSync()[0]
        let p2 = (prediction * 100) + ""
        array.push(parseInt(p2.substring(0, 2)))
    }

    const result = [...array.reduce((r, n) => // create a map of occurrences
        r.set(n, (r.get(n) || 0) + 1), new Map()
    )]
        .reduce((r, v) => v[1] < r[1] ? v : r)[0]; // get the the item that appear less times


    console.log(temp);
    document.getElementById("output").innerHTML = result

}