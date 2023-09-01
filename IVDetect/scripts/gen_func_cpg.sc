def getAST(cpgFile: String, funcName: String, outFile: String) = {
  importCpg(cpgFile)
  cpg.method(funcName).dotAst.l |> outFile
}

def getCPG(cpgFile: String, funcName: String, outFile: String) = {
  importCpg(cpgFile)
  cpg.method(funcName).dotCpg14.l |> outFile
}


@main def exec(cpgFile: String, funcName: String) = {
  importCpg(cpgFile)
  cpg.method(funcName).dotAst.l |> "ast.dot"
  cpg.method(funcName).dotCpg14.l |> "cpg.dot"
}