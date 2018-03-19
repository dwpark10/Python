

class AddLayer:
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout

class MulLayer:
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x

def apple_net():
    apple_price = 100
    apple_count = 2
    tax = 1.1

    layer_apple = MulLayer()
    layer_tax = MulLayer()

    # forward
    apple_total = layer_apple.forward(apple_price, apple_count)
    price = layer_tax.forward(apple_total, tax)

    print(' forward :', apple_total, price)

    # backward
    d_price = 1
    d_apple_total, d_apple_tax = layer_tax.backward(d_price)
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)

    print(' backword :', d_apple_price, d_apple_count, d_apple_tax)

def fruit_net():
    apple_price = 100
    apple_count = 2
    mango_price = 150
    mango_count = 3
    tax = 1.1

    layer_apple = MulLayer()
    layer_mango = MulLayer()
    layer_fruit = AddLayer()
    layer_tax = MulLayer()

    # forward
    apple_total = layer_apple.forward(apple_price, apple_count)
    mango_total = layer_mango.forward(mango_price, mango_count)
    fruit_total = layer_fruit.forward(apple_total, mango_total)
    price = layer_tax.forward(fruit_total, tax)

    print(' forward :', price)

    # backward
    d_price = 1
    d_fruit_total, d_fruit_tax = layer_tax.backward(d_price)
    d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
    d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)

    print(' backword :', d_apple_price, d_apple_count, d_mango_price, d_mango_count, d_fruit_tax)

def back_propagation(apple_price, mango_price):
    apple_count = 2
    mango_count = 3
    tax = 1.1

    target = 715
    # weight
    # apple_price = 100
    # mango_price = 150

    layer_apple = MulLayer()
    layer_mango = MulLayer()
    layer_fruit = AddLayer()
    layer_tax = MulLayer()

    for i in range(200):
        # forward
        apple_total = layer_apple.forward(apple_price, apple_count)
        mango_total = layer_mango.forward(mango_price, mango_count)
        fruit_total = layer_fruit.forward(apple_total, mango_total)
        price = layer_tax.forward(fruit_total, tax)

        # print(' forward :', price)

        # backward
        d_price = (price - target)
        d_fruit_total, d_fruit_tax = layer_tax.backward(d_price)
        d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
        d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)
        d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)

        # print(' backword :', d_apple_price, d_apple_count, d_mango_price, d_mango_count, d_fruit_tax)

        apple_price -= 0.005 * d_apple_price
        mango_price -= 0.005 * d_mango_price

    print(apple_price, mango_price)



# apple_net()
# fruit_net()
back_propagation(10, 15)
back_propagation(-10, -15)
back_propagation(10, -15)
back_propagation(-10, 15)